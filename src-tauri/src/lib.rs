use serde::Serialize;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, State};

#[derive(Clone, Default, Serialize)]
struct BackendStatus {
    started: bool,
    error: Option<String>,
    app_root: String,
    python: String,
    script: String,
}

struct PythonServer {
    child: Mutex<Option<Child>>,
    status: Mutex<BackendStatus>,
}

#[tauri::command]
fn get_api_url() -> String {
    "http://127.0.0.1:7878".to_string()
}

#[tauri::command]
fn get_backend_status(state: State<PythonServer>) -> BackendStatus {
    state.status.lock().unwrap().clone()
}

#[tauri::command]
fn exit_app(app: tauri::AppHandle, state: State<PythonServer>) -> Result<(), String> {
    if let Some(mut child) = state.child.lock().map_err(|err| err.to_string())?.take() {
        let _ = child.kill();
        let _ = child.wait();
        println!("[bili-app] Python server stopped");
    }

    app.exit(0);
    Ok(())
}

fn find_app_root() -> PathBuf {
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("src-python").join("api_server.py").exists() {
            return cwd;
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(PathBuf::from).unwrap_or_default();
        for _ in 0..8 {
            if dir.join("src-python").join("api_server.py").exists() {
                return dir;
            }
            match dir.parent() {
                Some(parent) => dir = parent.to_path_buf(),
                None => break,
            }
        }
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn find_python(app_root: &PathBuf) -> String {
    if let Ok(py) = std::env::var("BILIRUBIN_PYTHON") {
        if !py.is_empty() {
            return py;
        }
    }

    let venv_names: &[&str] = if cfg!(windows) {
        &[".venv-bilirubin", ".venv-bili", ".venv", "venv", ".venv-lin"]
    } else {
        &[".venv-lin", ".venv-bilirubin", ".venv-bili", ".venv", "venv"]
    };

    if cfg!(windows) {
        for name in venv_names {
            let path = app_root.join(name).join("Scripts").join("python.exe");
            if path.exists() {
                return path.to_string_lossy().into_owned();
            }
        }
    } else {
        for name in venv_names {
            let path = app_root.join(name).join("bin").join("python3");
            if path.exists() {
                return path.to_string_lossy().into_owned();
            }
        }
    }

    if cfg!(windows) {
        "python".to_string()
    } else {
        "python3".to_string()
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(PythonServer {
            child: Mutex::new(None),
            status: Mutex::new(BackendStatus::default()),
        })
        .setup(|app| {
            let mut app_root = find_app_root();
            if !app_root.join("src-python").join("api_server.py").exists() {
                if let Ok(resource_dir) = app.path().resource_dir() {
                    if resource_dir.join("src-python").join("api_server.py").exists() {
                        app_root = resource_dir;
                    }
                }
            }
            let python = find_python(&app_root);
            let script = app_root.join("src-python").join("api_server.py");

            println!("[bili-app] app_root : {}", app_root.display());
            println!("[bili-app] python   : {python}");
            println!("[bili-app] script   : {}", script.display());

            let status_base = BackendStatus {
                started: false,
                error: None,
                app_root: app_root.to_string_lossy().into_owned(),
                python: python.clone(),
                script: script.to_string_lossy().into_owned(),
            };

            match Command::new(&python)
                .arg(&script)
                .current_dir(&app_root)
                .spawn()
            {
                Ok(child) => {
                    let state = app.state::<PythonServer>();
                    *state.child.lock().unwrap() = Some(child);
                    *state.status.lock().unwrap() = BackendStatus {
                        started: true,
                        ..status_base
                    };
                    println!("[bili-app] Python server started on 127.0.0.1:7878");
                }
                Err(err) => {
                    let state = app.state::<PythonServer>();
                    *state.status.lock().unwrap() = BackendStatus {
                        error: Some(err.to_string()),
                        ..status_base
                    };
                    eprintln!("[bili-app] Failed to start Python server: {err}");
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::Destroyed) {
                if let Some(state) = window.try_state::<PythonServer>() {
                    if let Some(mut child) = state.child.lock().unwrap().take() {
                        let _ = child.kill();
                        println!("[bili-app] Python server stopped");
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_api_url, get_backend_status, exit_app])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
