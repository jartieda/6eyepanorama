# Panorama Stitcher GUI - PySide6

Interfaz gráfica profesional para stitching de panoramas 360° desde cámaras fisheye.

## Características

- **Stitcher**: Procesa 6 imágenes fisheye y genera panoramas 360°
  - Carga automática de configuración JSON
  - Visualización de imágenes de entrada
  - Vista del panorama resultante
  - Muestra pasos intermedios del proceso
  - Soporte para procesamiento iterativo de videos (futuro)

- **Calibration**: Calibración de cámaras omnidireccionales
  - Detección automática de tableros de ajedrez
  - Detección de círculos fisheye
  - Cálculo de matrices de cámara (K, D, xi)
  - Visualización en tiempo real del proceso

- **Settings**: Editor de configuración JSON
  - Edición manual de parámetros de cámara
  - Validación de sintaxis JSON
  - Guardado seguro con verificación

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

## Estructura

```
sticher_gui_qt/
├── main.py                 # Aplicación principal
├── views/                  # Vistas de la UI
│   ├── stitcher_view.py    # Tab de stitching
│   ├── calibration_view.py # Tab de calibración
│   └── settings_view.py    # Tab de configuración
└── logic/                  # Lógica de negocio
    ├── stitcher_logic.py   # Importa PanoramaStitcher
    └── calibration_logic.py# Importa funciones de calibración
```

## Dependencias

- PySide6: Framework Qt para Python
- OpenCV: Procesamiento de imágenes
- NumPy: Cálculos numéricos
- SciPy: Optimización y transformaciones
- PyTorch + torchvision: Optical flow (RAFT)

## Notas

La lógica de procesamiento reutiliza los scripts de `stich_old/`:
- `runstich.py`: Clase PanoramaStitcher completa
- `mycalibrate.py`: Funciones de calibración
