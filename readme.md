# 6 Eye Panorama - Stitching Project

## Descripción General
Proyecto de stitching para cámaras panorámicas de 6 ojos (360°). Incluye herramientas de calibración, procesamiento de imágenes y una interfaz gráfica de usuario desarrollada con PyQt.

## Estructura del Proyecto

```
6eyepanorama/
├── config_features.py          # Configuración de características del sistema
├── insta360.json               # Parámetros de calibración para cámara Insta360
├── kandao.json                 # Parámetros de calibración para cámara Kandao
├── mycalibrate.py              # Script de calibración de cámaras
├── runstich.py                 # Script principal de stitching
├── requirements.txt            # Dependencias del proyecto
├── doc/                        # Documentación
│   ├── calibration.md          # Documentación de calibración
│   └── TECHNICAL_DOCUMENTATION.md
└── sticher_gui_qt/             # Interfaz gráfica con PyQt
    ├── main.py                 # Punto de entrada de la GUI
    ├── requirements.txt        # Dependencias de la GUI
    ├── logic/                  # Lógica de negocio
    │   ├── calibration_logic.py
    │   └── stitcher_logic.py
    └── views/                  # Vistas de la aplicación
        ├── calibration_view.py
        ├── panorama_viewer_360.py
        ├── settings_view.py
        └── stitcher_view.py
```

## Componentes Principales

### Scripts Base
- **runstich.py**: Script principal para procesamiento de imágenes panorámicas. Utiliza los parámetros de calibración de los archivos JSON.
- **mycalibrate.py**: Herramienta de calibración de cámaras para ajustar los parámetros de stitching.
- **config_features.py**: Configuración de características y parámetros del sistema.

### Archivos de Configuración
- **kandao.json**: Parámetros de calibración para cámara Kandao
- **insta360.json**: Parámetros de calibración para cámara Insta360

### Interfaz Gráfica (sticher_gui_qt)
Aplicación PyQt que proporciona una interfaz visual para:
- Calibración de cámaras
- Procesamiento de imágenes (stitching)
- Visualización de panoramas 360°
- Configuración de parámetros

**Estructura:**
- `main.py`: Punto de entrada de la aplicación
- `logic/`: Módulos de lógica de negocio
  - `calibration_logic.py`: Lógica de calibración
  - `stitcher_logic.py`: Lógica de stitching
- `views/`: Componentes de interfaz de usuario
  - `calibration_view.py`: Vista de calibración
  - `stitcher_view.py`: Vista de stitching
  - `settings_view.py`: Vista de configuración
  - `panorama_viewer_360.py`: Visor panorámico 360°

### Documentación
- **doc/calibration.md**: Guía detallada del proceso de calibración
- **doc/TECHNICAL_DOCUMENTATION.md**: Documentación técnica del proyecto

## Instalación

```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Para la interfaz gráfica
pip install -r sticher_gui_qt/requirements.txt
```

## Uso

### Script de Línea de Comandos

El script `runstich.py` acepta los siguientes parámetros:

```bash
python runstich.py --config <ruta_config> --input_template <plantilla_entrada> --output <ruta_salida> [--show]
```

**Parámetros:**
- `--config`: Ruta al archivo JSON de configuración (obligatorio)
- `--input_template`: Plantilla del nombre de archivo de entrada con `{id}` como placeholder (obligatorio)
- `--output`: Nombre del archivo de salida (obligatorio)
- `--show`: Mostrar pasos intermedios del procesamiento (opcional)

**Ejemplos:**

```bash
# Usando configuración de Kandao
python runstich.py --config kandao.json --input_template ../dataset/tuneladora/origin_{id}_1.jpg --output panorama_output.jpg

# Usando configuración de Insta360 y mostrando pasos
python runstich.py --config insta360.json --input_template ../dataset/peine/cam_{id}.jpg --output result.jpg --show

# Procesamiento de un conjunto específico de imágenes
python runstich.py --config kandao.json --input_template /path/to/images/frame_{id}.jpg --output /path/to/output/panorama.jpg
```

### Interfaz Gráfica
```bash
python sticher_gui_qt/main.py
```

### Calibración
```bash
python mycalibrate.py
```

## Documentación Adicional

Para más información sobre el proceso de calibración, consulta [doc/calibration.md](doc/calibration.md).

Para detalles técnicos del sistema, consulta [doc/TECHNICAL_DOCUMENTATION.md](doc/TECHNICAL_DOCUMENTATION.md).