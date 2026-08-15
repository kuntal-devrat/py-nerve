use enigo::{
    Axis, Button, Coordinate, Direction, Enigo, Keyboard, Mouse, Settings,
};
use image::{DynamicImage, GenericImageView};
use ocr_rs::{DetModel, RecModel};
use pyo3::prelude::*;
use std::io::Cursor;
use std::sync::Mutex;
use std::time::Instant;

/// A single OCR element: (text, confidence 0-1, (x, y, width, height)).
type OcrElement = (String, f64, (u32, u32, u32, u32));

/// Global OCR state: detection model + recognition model.
struct OcrState {
    det: DetModel,
    rec: RecModel,
}

static OCR_STATE: Mutex<Option<OcrState>> = Mutex::new(None);

fn runtime_err(msg: impl Into<String>) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(msg.into())
}

fn value_err(msg: impl Into<String>) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

/// Capture a monitor (optionally a sub-region) as a DynamicImage.
fn capture_image(
    region: Option<(u32, u32, u32, u32)>,
    monitor_index: Option<usize>,
) -> Result<DynamicImage, String> {
    let monitors =
        xcap::Monitor::all().map_err(|e| format!("Failed to list monitors: {e}"))?;

    let monitor = if let Some(idx) = monitor_index {
        monitors.get(idx).ok_or_else(|| {
            format!("Monitor index {idx} out of range (found {} monitors)", monitors.len())
        })?
    } else {
        monitors
            .iter()
            .find(|m| m.is_primary().unwrap_or(false))
            .or(monitors.first())
            .ok_or_else(|| "No monitors found".to_string())?
    };

    let rgba_image = monitor
        .capture_image()
        .map_err(|e| format!("Failed to capture screen: {e}"))?;

    let image = DynamicImage::ImageRgba8(rgba_image);

    let Some((x, y, w, h)) = region else {
        return Ok(image);
    };

    // Clamp the region to the image bounds (crop_imm panics out of bounds).
    let (img_w, img_h) = image.dimensions();
    let x = x.min(img_w);
    let y = y.min(img_h);
    let w = w.min(img_w - x);
    let h = h.min(img_h - y);
    if w == 0 || h == 0 {
        return Err(format!(
            "Invalid region ({x},{y},{w},{h}) for image of size ({img_w},{img_h})"
        ));
    }

    let cropped = image::imageops::crop_imm(&image, x, y, w, h).to_image();
    Ok(DynamicImage::ImageRgba8(cropped))
}

/// Run detection + batched recognition on an image.
fn run_ocr(state: &OcrState, img: &DynamicImage) -> Result<Vec<OcrElement>, String> {
    let detections = state
        .det
        .detect_and_crop(img)
        .map_err(|e| format!("Detection failed: {e}"))?;

    // Batch recognition: one MNN session resize per batch instead of per crop.
    let refs: Vec<&DynamicImage> = detections.iter().map(|(crop, _)| crop).collect();
    let rec_results = state
        .rec
        .recognize_batch_ref(&refs)
        .map_err(|e| format!("Recognition failed: {e}"))?;

    let mut results: Vec<OcrElement> = Vec::with_capacity(rec_results.len());
    for (rec_result, (_, textbox)) in rec_results.into_iter().zip(detections.iter()) {
        if rec_result.text.trim().is_empty() {
            continue;
        }
        let left = textbox.rect.left() as u32;
        let top = textbox.rect.top() as u32;
        let width = textbox.rect.width() as u32;
        let height = textbox.rect.height() as u32;
        results.push((
            rec_result.text,
            rec_result.confidence as f64,
            (left, top, width, height),
        ));
    }
    Ok(results)
}

/// Capture a screenshot and return PNG bytes.
///
/// Args:
///     region: Optional (x, y, width, height) tuple. If None, captures full primary screen.
///     monitor_index: Optional zero-based monitor index.
///
/// Returns:
///     PNG image bytes.
#[pyfunction]
#[pyo3(signature = (region=None, monitor_index=None))]
fn screenshot(
    region: Option<(u32, u32, u32, u32)>,
    monitor_index: Option<usize>,
) -> PyResult<Vec<u8>> {
    let final_image = capture_image(region, monitor_index).map_err(runtime_err)?;

    let mut png_bytes: Vec<u8> = Vec::new();
    final_image
        .write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .map_err(|e| runtime_err(format!("Failed to encode PNG: {e}")))?;

    Ok(png_bytes)
}

/// Capture the screen (or region) and run OCR in one native pass.
///
/// Unlike `screenshot()` + `ocr_from_png_bytes()`, this never leaves Rust,
/// so it skips the PNG encode -> decode -> re-encode round-trips.
///
/// Args:
///     region: Optional (x, y, width, height) tuple.
///     monitor_index: Optional zero-based monitor index.
///
/// Returns:
///     List of tuples: [(text, confidence, (x, y, w, h)), ...]
#[pyfunction]
#[pyo3(signature = (region=None, monitor_index=None))]
fn capture_ocr(
    py: Python<'_>,
    region: Option<(u32, u32, u32, u32)>,
    monitor_index: Option<usize>,
) -> PyResult<Vec<OcrElement>> {
    py.detach(|| {
        let img = capture_image(region, monitor_index)?;
        let guard = OCR_STATE.lock().map_err(|_| "OCR lock poisoned".to_string())?;
        let state = guard.as_ref().ok_or_else(|| {
            "OCR engine not initialized. Call init_ocr() first.".to_string()
        })?;
        let mut results = run_ocr(state, &img)?;
        if let Some((rx, ry, _, _)) = region {
            for (_text, _conf, (x, y, _w, _h)) in results.iter_mut() {
                *x += rx;
                *y += ry;
            }
        }
        Ok::<_, String>(results)
    })
    .map_err(runtime_err)
}

/// Run OCR on PNG image bytes.
///
/// Args:
///     png_bytes: PNG-encoded image bytes.
///
/// Returns:
///     List of tuples: [(text, confidence, (x, y, w, h)), ...]
#[pyfunction]
fn ocr_from_png_bytes(py: Python<'_>, png_bytes: Vec<u8>) -> PyResult<Vec<OcrElement>> {
    py.detach(|| {
        let img = image::load_from_memory(&png_bytes)
            .map_err(|e| format!("Failed to decode image: {e}"))?;
        let guard = OCR_STATE.lock().map_err(|_| "OCR lock poisoned".to_string())?;
        let state = guard.as_ref().ok_or_else(|| {
            "OCR engine not initialized. Call init_ocr() first.".to_string()
        })?;
        run_ocr(state, &img)
    })
    .map_err(runtime_err)
}

/// Capture the screen and compute a cheap perceptual hash of it.
///
/// Two screenshots that produce the same hash are (almost certainly)
/// pixel-identical, so callers can skip re-running OCR when the hash
/// hasn't changed. ~8x8 grayscale average-hash, sub-millisecond.
///
/// Args:
///     region: Optional (x, y, width, height) tuple.
///     monitor_index: Optional zero-based monitor index.
///
/// Returns:
///     u64 hash of the captured region.
#[pyfunction]
#[pyo3(signature = (region=None, monitor_index=None))]
fn capture_hash(
    py: Python<'_>,
    region: Option<(u32, u32, u32, u32)>,
    monitor_index: Option<usize>,
) -> PyResult<u64> {
    py.detach(|| {
        let img = capture_image(region, monitor_index)?;
        let small = img.thumbnail_exact(8, 8);
        let gray = image::imageops::grayscale(&small);
        let raw = gray.as_raw();
        let mean = raw.iter().map(|&p| p as u32).sum::<u32>() / raw.len().max(1) as u32;

        let mut hash: u64 = 0;
        for (i, &p) in raw.iter().enumerate() {
            if (p as u32) >= mean {
                hash |= 1u64 << (i % 64);
            }
        }
        Ok::<u64, String>(hash)
    })
    .map_err(runtime_err)
}

/// List all connected monitors with their (index, name, is_primary, (x, y, width, height)).
#[pyfunction]
fn list_monitors() -> PyResult<Vec<(usize, String, bool, (i32, i32, u32, u32))>> {
    let monitors =
        xcap::Monitor::all().map_err(|e| runtime_err(format!("Failed to list monitors: {e}")))?;
    let mut res = Vec::with_capacity(monitors.len());
    for (idx, m) in monitors.iter().enumerate() {
        let name = m.name().unwrap_or_else(|_| format!("Monitor {idx}"));
        let is_primary = m.is_primary().unwrap_or(false);
        let x = m.x().unwrap_or(0);
        let y = m.y().unwrap_or(0);
        let width = m.width().unwrap_or(0);
        let height = m.height().unwrap_or(0);
        res.push((idx, name, is_primary, (x, y, width, height)));
    }
    Ok(res)
}

/// Read text from the system clipboard.
#[pyfunction]
fn get_clipboard() -> PyResult<String> {
    let mut clipboard = arboard::Clipboard::new()
        .map_err(|e| runtime_err(format!("Failed to initialize clipboard: {e}")))?;
    clipboard
        .get_text()
        .map_err(|e| runtime_err(format!("Failed to read clipboard text: {e}")))
}

/// Write text to the system clipboard.
#[pyfunction]
fn set_clipboard(text: String) -> PyResult<()> {
    let mut clipboard = arboard::Clipboard::new()
        .map_err(|e| runtime_err(format!("Failed to initialize clipboard: {e}")))?;
    clipboard
        .set_text(text)
        .map_err(|e| runtime_err(format!("Failed to write clipboard text: {e}")))
}

/// Run `f` with a lazily-created thread-local Enigo instance.
fn with_enigo<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut Enigo) -> Result<R, String>,
{
    thread_local! {
        static ENIGO: std::cell::RefCell<Option<Enigo>> = std::cell::RefCell::new(None);
    }
    ENIGO.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            let enigo = Enigo::new(&Settings::default())
                .map_err(|e| format!("Failed to init enigo: {e}"))?;
            *borrow = Some(enigo);
        }
        f(borrow.as_mut().unwrap())
    })
}

/// Get the current mouse cursor position.
///
/// Returns:
///     Tuple of (x, y) coordinates.
#[pyfunction]
fn get_mouse_position() -> PyResult<(f64, f64)> {
    with_enigo(|enigo| {
        let pos = enigo
            .location()
            .map_err(|e| format!("Failed to get mouse position: {e}"))?;
        Ok((pos.0 as f64, pos.1 as f64))
    })
    .map_err(runtime_err)
}

/// Instant mouse move to absolute coordinates.
#[pyfunction]
fn move_to(x: f64, y: f64) -> PyResult<()> {
    with_enigo(|enigo| {
        enigo
            .move_mouse(x as i32, y as i32, Coordinate::Abs)
            .map_err(|e| format!("Failed to move mouse: {e}"))?;
        Ok(())
    })
    .map_err(runtime_err)
}

fn bezier_move_impl(enigo: &mut Enigo, x: f64, y: f64, duration: f64) -> Result<bool, String> {
    let target_x = x;
    let target_y = y;

    let start = enigo
        .location()
        .map_err(|e| format!("Failed to get mouse position: {e}"))?;

    let mut start_x = start.0 as f64;
    let mut start_y = start.1 as f64;
    let mut was_interrupted_overall = false;

    loop {
        let mid_x = (start_x + target_x) / 2.0;
        let mid_y = (start_y + target_y) / 2.0;

        let distance = ((target_x - start_x).powi(2) + (target_y - start_y).powi(2)).sqrt();
        if distance < 2.0 {
            break;
        }

        // Offset control points based on distance with randomness
        let offset_factor = (distance / 200.0).min(3.0);
        let cp1_x = mid_x + (fastrand::f64() - 0.5) * 100.0 * offset_factor;
        let cp1_y = mid_y + (fastrand::f64() - 0.5) * 100.0 * offset_factor;
        let cp2_x = mid_x + (fastrand::f64() - 0.5) * 80.0 * offset_factor;
        let cp2_y = mid_y + (fastrand::f64() - 0.5) * 80.0 * offset_factor;

        // Number of steps based on distance
        let steps = ((distance / 5.0) as u32).max(30).min(80);
        let step_duration = duration / steps as f64;

        let start_time = Instant::now();

        let mut last_expected_x = start_x;
        let mut last_expected_y = start_y;
        let mut interrupted = false;

        for i in 1..=steps {
            // Check if human has moved the mouse (interference detection)
            let current_pos = enigo
                .location()
                .map_err(|e| format!("Failed to get mouse position: {e}"))?;
            let current_x = current_pos.0 as f64;
            let current_y = current_pos.1 as f64;

            let diff_x = current_x - last_expected_x;
            let diff_y = current_y - last_expected_y;
            let delta = (diff_x.powi(2) + diff_y.powi(2)).sqrt();

            // Threshold of 15 pixels to trigger pause-and-resume
            if delta > 15.0 {
                interrupted = true;
                was_interrupted_overall = true;
                break;
            }

            let t = i as f64 / steps as f64;

            // Cubic bezier interpolation
            let u = 1.0 - t;
            let px = u.powi(3) * start_x
                + 3.0 * u.powi(2) * t * cp1_x
                + 3.0 * u * t.powi(2) * cp2_x
                + t.powi(3) * target_x;
            let py = u.powi(3) * start_y
                + 3.0 * u.powi(2) * t * cp1_y
                + 3.0 * u * t.powi(2) * cp2_y
                + t.powi(3) * target_y;

            enigo
                .move_mouse(px as i32, py as i32, Coordinate::Abs)
                .map_err(|e| format!("Failed to move mouse: {e}"))?;

            last_expected_x = px;
            last_expected_y = py;

            // Sleep to maintain duration
            let elapsed = start_time.elapsed().as_secs_f64();
            let target_time = step_duration * i as f64;
            if elapsed < target_time {
                let sleep_ms = ((target_time - elapsed) * 1000.0) as u64;
                std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
            }
        }

        if interrupted {
            // Wait until the mouse becomes static for 800ms
            let mut static_ms = 0;
            let mut last_pos = enigo
                .location()
                .map_err(|e| format!("Failed to get mouse position: {e}"))?;

            while static_ms < 800 {
                std::thread::sleep(std::time::Duration::from_millis(50));
                let current_pos = enigo
                    .location()
                    .map_err(|e| format!("Failed to get mouse position: {e}"))?;

                let dx = current_pos.0 as f64 - last_pos.0 as f64;
                let dy = current_pos.1 as f64 - last_pos.1 as f64;
                let dist = (dx.powi(2) + dy.powi(2)).sqrt();

                if dist <= 3.0 {
                    static_ms += 50;
                } else {
                    static_ms = 0;
                    last_pos = current_pos;
                }
            }

            // Reset start position to current mouse position and recalculate the curve
            let current_pos = enigo
                .location()
                .map_err(|e| format!("Failed to get mouse position: {e}"))?;
            start_x = current_pos.0 as f64;
            start_y = current_pos.1 as f64;
        } else {
            break;
        }
    }

    Ok(was_interrupted_overall)
}

/// Move mouse along a cubic bezier curve with human-like motion.
///
/// Args:
///     x: Target X coordinate.
///     y: Target Y coordinate.
///     duration: Movement duration in seconds (0.2 - 0.5 recommended).
///
/// Returns:
///     True if human interference was detected and recovered from, False otherwise.
#[pyfunction]
#[pyo3(signature = (x, y, duration=0.4))]
fn bezier_move(py: Python<'_>, x: f64, y: f64, duration: f64) -> PyResult<bool> {
    // The movement takes `duration` seconds of sleeping; don't hold the GIL.
    py.detach(|| with_enigo(|enigo| bezier_move_impl(enigo, x, y, duration)))
        .map_err(runtime_err)
}


/// Perform a mouse click.
///
/// Args:
///     button: "left", "right", or "middle". Default is "left".
#[pyfunction]
#[pyo3(signature = (button="left"))]
fn click(button: &str) -> PyResult<()> {
    let btn = parse_button(button).map_err(value_err)?;
    with_enigo(|enigo| {
        enigo
            .button(btn, Direction::Click)
            .map_err(|e| format!("Failed to click: {e}"))?;
        Ok(())
    })
    .map_err(runtime_err)
}

/// Perform a double click.
#[pyfunction]
fn double_click(py: Python<'_>) -> PyResult<()> {
    py.detach(|| {
        with_enigo(|enigo| {
            enigo
                .button(Button::Left, Direction::Click)
                .map_err(|e| format!("Failed to double click: {e}"))?;
            std::thread::sleep(std::time::Duration::from_millis(50));
            enigo
                .button(Button::Left, Direction::Click)
                .map_err(|e| format!("Failed to double click: {e}"))?;
            Ok(())
        })
    })
    .map_err(runtime_err)
}

/// Press a mouse button down without releasing it.
#[pyfunction]
#[pyo3(signature = (button="left"))]
fn mouse_down(button: &str) -> PyResult<()> {
    let btn = parse_button(button).map_err(value_err)?;
    with_enigo(|enigo| {
        enigo
            .button(btn, Direction::Press)
            .map_err(|e| format!("Failed to press mouse button: {e}"))?;
        Ok(())
    })
    .map_err(runtime_err)
}

/// Release a previously pressed mouse button.
#[pyfunction]
#[pyo3(signature = (button="left"))]
fn mouse_up(button: &str) -> PyResult<()> {
    let btn = parse_button(button).map_err(value_err)?;
    with_enigo(|enigo| {
        enigo
            .button(btn, Direction::Release)
            .map_err(|e| format!("Failed to release mouse button: {e}"))?;
        Ok(())
    })
    .map_err(runtime_err)
}

/// Scroll the mouse wheel. Positive scrolls up, negative scrolls down.
///
/// Args:
///     amount: Number of wheel notches. Positive = up (away from user).
///     axis: "vertical" (default) or "horizontal".
#[pyfunction]
#[pyo3(signature = (amount, axis="vertical"))]
fn scroll(amount: i32, axis: &str) -> PyResult<()> {
    with_enigo(|enigo| {
        let enigo_axis = match axis {
            "horizontal" => Axis::Horizontal,
            _ => Axis::Vertical,
        };
        // enigo sends -length * WHEEL_DELTA for vertical, so negate to keep
        // our API convention (positive = scroll up).
        enigo
            .scroll(-amount, enigo_axis)
            .map_err(|e| format!("Failed to scroll: {e}"))?;
        Ok(())
    })
    .map_err(runtime_err)
}

/// Type text string character by character with human interference awareness.
///
/// Args:
///     text: The text to type.
///     interval: Delay between characters in seconds. Default is 0.05.
#[pyfunction]
#[pyo3(signature = (text, interval=0.05))]
fn type_text(py: Python<'_>, text: &str, interval: f64) -> PyResult<()> {
    py.detach(|| {
        with_enigo(|enigo| {
            let sleep_duration = std::time::Duration::from_millis((interval * 1000.0) as u64);
            let mut last_mouse = enigo
                .location()
                .map_err(|e| format!("Failed to get mouse position: {e}"))?;

            for ch in text.chars() {
                // Check if human moved the mouse while typing
                let current_mouse = enigo
                    .location()
                    .map_err(|e| format!("Failed to get mouse position: {e}"))?;
                let dx = (current_mouse.0 - last_mouse.0) as f64;
                let dy = (current_mouse.1 - last_mouse.1) as f64;
                if (dx * dx + dy * dy).sqrt() > 20.0 {
                    // Human moved mouse: pause typing until mouse is released and static for 600ms
                    let mut static_ms = 0;
                    let mut pause_pos = current_mouse;
                    while static_ms < 600 {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        let poll_pos = enigo
                            .location()
                            .map_err(|e| format!("Failed to get mouse position: {e}"))?;
                        let pdx = (poll_pos.0 - pause_pos.0) as f64;
                        let pdy = (poll_pos.1 - pause_pos.1) as f64;
                        if (pdx * pdx + pdy * pdy).sqrt() <= 3.0 {
                            static_ms += 50;
                        } else {
                            static_ms = 0;
                            pause_pos = poll_pos;
                        }
                    }
                    last_mouse = enigo
                        .location()
                        .map_err(|e| format!("Failed to get mouse position: {e}"))?;
                }

                if ch == '\n' || ch == '\r' {
                    enigo
                        .key(enigo::Key::Return, Direction::Click)
                        .map_err(|e| format!("Failed to type Return: {e}"))?;
                } else if ch == '\t' {
                    enigo
                        .key(enigo::Key::Tab, Direction::Click)
                        .map_err(|e| format!("Failed to type Tab: {e}"))?;
                } else {
                    enigo
                        .text(&ch.to_string())
                        .map_err(|e| format!("Failed to type character: {e}"))?;
                }

                if !sleep_duration.is_zero() {
                    std::thread::sleep(sleep_duration);
                }
            }

            Ok(())
        })
    })
    .map_err(runtime_err)
}


/// Press a single key.
///
/// Args:
///     key: Key name (e.g., "enter", "tab", "escape", "backspace", "space",
///          "up", "down", "left", "right", "home", "end", "f1"-"f12", etc.)
#[pyfunction]
fn press_key(key: &str) -> PyResult<()> {
    let enigo_key =
        parse_key(key).ok_or_else(|| value_err(format!("Unknown key: {key}")))?;
    with_enigo(|enigo| {
        enigo
            .key(enigo_key, Direction::Click)
            .map_err(|e| format!("Failed to press key: {e}"))?;
        Ok(())
    })
    .map_err(runtime_err)
}

/// Press a key combination (e.g., Ctrl+S, Alt+Tab).
///
/// Args:
///     keys: List of key names to press together. Last key is released first.
///           Example: ["ctrl", "s"] for Ctrl+S.
#[pyfunction]
fn key_combo(keys: Vec<String>) -> PyResult<()> {
    if keys.is_empty() {
        return Err(value_err("keys list cannot be empty"));
    }

    let mut enigo_keys: Vec<enigo::Key> = Vec::with_capacity(keys.len());
    for key_str in &keys {
        let k = parse_key(key_str)
            .ok_or_else(|| value_err(format!("Unknown key: {key_str}")))?;
        enigo_keys.push(k);
    }

    with_enigo(|enigo| {
        // Hold all keys down
        for k in &enigo_keys {
            enigo
                .key(*k, Direction::Press)
                .map_err(|e| format!("Failed to press key: {e}"))?;
        }

        // Release in reverse order
        for k in enigo_keys.iter().rev() {
            enigo
                .key(*k, Direction::Release)
                .map_err(|e| format!("Failed to release key: {e}"))?;
        }

        Ok(())
    })
    .map_err(runtime_err)
}

/// Initialize the Rust OCR engine with PP-OCRv5 English models.
///
/// Args:
///     det_model_path: Path to the detection MNN model (e.g. "PP-OCRv5_mobile_det.mnn").
///     rec_model_path: Path to the recognition MNN model (e.g. "en_PP-OCRv5_mobile_rec_infer.mnn").
///     keys_path: Path to the character set file (e.g. "ppocr_keys_en.txt").
///
/// Returns:
///     True if initialization succeeded.
#[pyfunction]
fn init_ocr(det_model_path: &str, rec_model_path: &str, keys_path: &str) -> PyResult<bool> {
    let det = DetModel::from_file(det_model_path, None)
        .map_err(|e| runtime_err(format!("Failed to load det model: {e}")))?;

    let rec = RecModel::from_file(rec_model_path, keys_path, None)
        .map_err(|e| runtime_err(format!("Failed to load rec model: {e}")))?;

    let mut state = OCR_STATE
        .lock()
        .map_err(|e| runtime_err(format!("Lock poisoned: {e}")))?;

    *state = Some(OcrState { det, rec });

    Ok(true)
}

fn parse_button(button: &str) -> Result<Button, String> {
    match button {
        "left" => Ok(Button::Left),
        "right" => Ok(Button::Right),
        "middle" => Ok(Button::Middle),
        other => Err(format!("Invalid button: {other}")),
    }
}

/// Parse a key string into an enigo Key.
fn parse_key(key: &str) -> Option<enigo::Key> {
    match key.to_lowercase().as_str() {
        "enter" | "return" => Some(enigo::Key::Return),
        "tab" => Some(enigo::Key::Tab),
        "escape" | "esc" => Some(enigo::Key::Escape),
        "backspace" => Some(enigo::Key::Backspace),
        "space" => Some(enigo::Key::Space),
        "delete" | "del" => Some(enigo::Key::Delete),
        "home" => Some(enigo::Key::Home),
        "end" => Some(enigo::Key::End),
        "pageup" | "page_up" => Some(enigo::Key::PageUp),
        "pagedown" | "page_down" => Some(enigo::Key::PageDown),
        "up" => Some(enigo::Key::UpArrow),
        "down" => Some(enigo::Key::DownArrow),
        "left" => Some(enigo::Key::LeftArrow),
        "right" => Some(enigo::Key::RightArrow),
        "capslock" | "caps_lock" => Some(enigo::Key::CapsLock),
        #[cfg(not(target_os = "macos"))]
        "numlock" | "num_lock" => Some(enigo::Key::Numlock),
        "scrolllock" | "scroll_lock" => Some(enigo::Key::Other(145)),
        #[cfg(not(target_os = "macos"))]
        "printscreen" | "print_screen" => Some(enigo::Key::Print),
        #[cfg(not(target_os = "macos"))]
        "insert" | "ins" => Some(enigo::Key::Insert),

        // Function keys
        "f1" => Some(enigo::Key::F1),
        "f2" => Some(enigo::Key::F2),
        "f3" => Some(enigo::Key::F3),
        "f4" => Some(enigo::Key::F4),
        "f5" => Some(enigo::Key::F5),
        "f6" => Some(enigo::Key::F6),
        "f7" => Some(enigo::Key::F7),
        "f8" => Some(enigo::Key::F8),
        "f9" => Some(enigo::Key::F9),
        "f10" => Some(enigo::Key::F10),
        "f11" => Some(enigo::Key::F11),
        "f12" => Some(enigo::Key::F12),

        // Modifiers
        "ctrl" | "control" => Some(enigo::Key::Control),
        "shift" => Some(enigo::Key::Shift),
        "alt" | "option" => Some(enigo::Key::Alt),
        "meta" | "cmd" | "command" | "win" | "super" => Some(enigo::Key::Meta),

        // Letters a-z
        "a" => Some(enigo::Key::Unicode('a')),
        "b" => Some(enigo::Key::Unicode('b')),
        "c" => Some(enigo::Key::Unicode('c')),
        "d" => Some(enigo::Key::Unicode('d')),
        "e" => Some(enigo::Key::Unicode('e')),
        "f" => Some(enigo::Key::Unicode('f')),
        "g" => Some(enigo::Key::Unicode('g')),
        "h" => Some(enigo::Key::Unicode('h')),
        "i" => Some(enigo::Key::Unicode('i')),
        "j" => Some(enigo::Key::Unicode('j')),
        "k" => Some(enigo::Key::Unicode('k')),
        "l" => Some(enigo::Key::Unicode('l')),
        "m" => Some(enigo::Key::Unicode('m')),
        "n" => Some(enigo::Key::Unicode('n')),
        "o" => Some(enigo::Key::Unicode('o')),
        "p" => Some(enigo::Key::Unicode('p')),
        "q" => Some(enigo::Key::Unicode('q')),
        "r" => Some(enigo::Key::Unicode('r')),
        "s" => Some(enigo::Key::Unicode('s')),
        "t" => Some(enigo::Key::Unicode('t')),
        "u" => Some(enigo::Key::Unicode('u')),
        "v" => Some(enigo::Key::Unicode('v')),
        "w" => Some(enigo::Key::Unicode('w')),
        "x" => Some(enigo::Key::Unicode('x')),
        "y" => Some(enigo::Key::Unicode('y')),
        "z" => Some(enigo::Key::Unicode('z')),

        // Numbers 0-9
        "0" => Some(enigo::Key::Unicode('0')),
        "1" => Some(enigo::Key::Unicode('1')),
        "2" => Some(enigo::Key::Unicode('2')),
        "3" => Some(enigo::Key::Unicode('3')),
        "4" => Some(enigo::Key::Unicode('4')),
        "5" => Some(enigo::Key::Unicode('5')),
        "6" => Some(enigo::Key::Unicode('6')),
        "7" => Some(enigo::Key::Unicode('7')),
        "8" => Some(enigo::Key::Unicode('8')),
        "9" => Some(enigo::Key::Unicode('9')),

        // Punctuation and symbols
        "-" | "dash" | "minus" => Some(enigo::Key::Unicode('-')),
        "=" | "equals" => Some(enigo::Key::Unicode('=')),
        "+" | "plus" => Some(enigo::Key::Unicode('+')),
        "_" | "underscore" => Some(enigo::Key::Unicode('_')),
        "!" | "exclamation" => Some(enigo::Key::Unicode('!')),
        "@" | "at" => Some(enigo::Key::Unicode('@')),
        "#" | "hash" | "numbersign" => Some(enigo::Key::Unicode('#')),
        "$" | "dollar" => Some(enigo::Key::Unicode('$')),
        "%" | "percent" => Some(enigo::Key::Unicode('%')),
        "^" | "caret" => Some(enigo::Key::Unicode('^')),
        "&" | "ampersand" => Some(enigo::Key::Unicode('&')),
        "*" | "asterisk" | "star" => Some(enigo::Key::Unicode('*')),
        "(" | "leftparen" | "left_paren" | "openparen" => Some(enigo::Key::Unicode('(')),
        ")" | "rightparen" | "right_paren" | "closeparen" => Some(enigo::Key::Unicode(')')),
        "{" | "leftbrace" | "left_brace" | "openbrace" => Some(enigo::Key::Unicode('{')),
        "}" | "rightbrace" | "right_brace" | "closebrace" => Some(enigo::Key::Unicode('}')),
        "|" | "pipe" | "bar" => Some(enigo::Key::Unicode('|')),
        ":" | "colon" => Some(enigo::Key::Unicode(':')),
        "\"" | "doublequote" | "double_quote" => Some(enigo::Key::Unicode('"')),
        "<" | "less" | "lessthan" | "less_than" => Some(enigo::Key::Unicode('<')),
        ">" | "greater" | "greaterthan" | "greater_than" => Some(enigo::Key::Unicode('>')),
        "?" | "question" | "questionmark" | "question_mark" => Some(enigo::Key::Unicode('?')),
        "~" | "tilde" => Some(enigo::Key::Unicode('~')),
        "[" | "leftbracket" | "left_bracket" => Some(enigo::Key::Unicode('[')),
        "]" | "rightbracket" | "right_bracket" => Some(enigo::Key::Unicode(']')),
        ";" | "semicolon" => Some(enigo::Key::Unicode(';')),
        "'" | "quote" | "apostrophe" => Some(enigo::Key::Unicode('\'')),
        "," | "comma" => Some(enigo::Key::Unicode(',')),
        "." | "period" | "dot" => Some(enigo::Key::Unicode('.')),
        "/" | "slash" | "forwardslash" | "forward_slash" => Some(enigo::Key::Unicode('/')),
        "\\" | "backslash" => Some(enigo::Key::Unicode('\\')),
        "`" | "backtick" | "grave" => Some(enigo::Key::Unicode('`')),

        // Numpad
        "numpad0" | "num0" => Some(enigo::Key::Unicode('0')),
        "numpad1" | "num1" => Some(enigo::Key::Unicode('1')),
        "numpad2" | "num2" => Some(enigo::Key::Unicode('2')),
        "numpad3" | "num3" => Some(enigo::Key::Unicode('3')),
        "numpad4" | "num4" => Some(enigo::Key::Unicode('4')),
        "numpad5" | "num5" => Some(enigo::Key::Unicode('5')),
        "numpad6" | "num6" => Some(enigo::Key::Unicode('6')),
        "numpad7" | "num7" => Some(enigo::Key::Unicode('7')),
        "numpad8" | "num8" => Some(enigo::Key::Unicode('8')),
        "numpad9" | "num9" => Some(enigo::Key::Unicode('9')),
        "numpad_add" | "numadd" => Some(enigo::Key::Unicode('+')),
        "numpad_subtract" | "numsub" => Some(enigo::Key::Unicode('-')),
        "numpad_multiply" | "nummul" => Some(enigo::Key::Unicode('*')),
        "numpad_divide" | "numdiv" => Some(enigo::Key::Unicode('/')),
        "numpad_decimal" | "numdec" => Some(enigo::Key::Unicode('.')),
        "numpad_enter" | "numenter" => Some(enigo::Key::Return),

        _ => None,
    }
}

/// PyO3 module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(screenshot, m)?)?;
    m.add_function(wrap_pyfunction!(capture_ocr, m)?)?;
    m.add_function(wrap_pyfunction!(capture_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ocr_from_png_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(list_monitors, m)?)?;
    m.add_function(wrap_pyfunction!(get_clipboard, m)?)?;
    m.add_function(wrap_pyfunction!(set_clipboard, m)?)?;
    m.add_function(wrap_pyfunction!(get_mouse_position, m)?)?;
    m.add_function(wrap_pyfunction!(move_to, m)?)?;
    m.add_function(wrap_pyfunction!(bezier_move, m)?)?;
    m.add_function(wrap_pyfunction!(click, m)?)?;
    m.add_function(wrap_pyfunction!(double_click, m)?)?;
    m.add_function(wrap_pyfunction!(mouse_down, m)?)?;
    m.add_function(wrap_pyfunction!(mouse_up, m)?)?;
    m.add_function(wrap_pyfunction!(scroll, m)?)?;
    m.add_function(wrap_pyfunction!(type_text, m)?)?;
    m.add_function(wrap_pyfunction!(press_key, m)?)?;
    m.add_function(wrap_pyfunction!(key_combo, m)?)?;
    m.add_function(wrap_pyfunction!(init_ocr, m)?)?;
    Ok(())
}
