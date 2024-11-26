FROM dolfinx/dolfinx:stable

# Actualizar el sistema e instalar dependencias necesarias
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libglx-mesa0 \
    xvfb \
    curl \
    bash-completion && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Instalar JupyterLab
RUN pip install --no-cache-dir jupyterlab ipywidgets pyvista ipyvtklink vtk ipywidgets

# Copiar el código de la aplicación al contenedor
COPY . /turbine_mesher

# Instalar el paquete local en modo editable
RUN pip install --no-cache-dir -e /turbine_mesher

# Exponer el puerto para Jupyter
EXPOSE 8888

WORKDIR /turbine_mesher/examples

ENV XDG_RUNTIME_DIR=/tmp/runtime-dir
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV LIBGL_ALWAYS_SOFTWARE=1
# Configurar el comando de inicio
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
