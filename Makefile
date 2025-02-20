# Makefile for compiling Python with full module support
# Includes dependencies for common optional modules and development tools

#-------------------------------------------------------------------------------
# Configuration Section
#-------------------------------------------------------------------------------

# Version configurations
BZIP2_VERSION      := 1.0.8
GDBM_VERSION       := 1.23
LIBFFI_VERSION     := 3.4.2
LIBUUID_VERSION    := 1.0.3
NCURSES_VERSION    := 6.3
OMPI_VERSION       := 4.1.8
OPENSSL_VERSION    := 3.4.1
PETSC_VERSION      := 3.22.3
PYTHON_VERSION     := 3.12.9
PRECICE_VERSION    := 3.1.2
READLINE_VERSION   := 8.2
SQLITE_VERSION     := 3490000
TCL_VERSION        := 8.6.13
TK_VERSION         := $(TCL_VERSION)
EIGEN_VERSION      := 3.3.7
BOOST_VERSION      := 1.68.0
LIBXML2_VERSION    := 2.13.6

# Dependency file names
PYTHON_TAR         := Python-$(PYTHON_VERSION).tgz
LIBFFI_TAR         := libffi-$(LIBFFI_VERSION).tar.gz
OPENSSL_TAR        := openssl-$(OPENSSL_VERSION).tar.gz
SQLITE_TAR         := sqlite-autoconf-$(SQLITE_VERSION).tar.gz
BZIP2_TAR          := bzip2-$(BZIP2_VERSION).tar.gz
NCURSES_TAR        := ncurses-$(NCURSES_VERSION).tar.gz
GDBM_TAR           := gdbm-$(GDBM_VERSION).tar.gz
TCL_TAR            := tcl$(TCL_VERSION)-src.tar.gz
TK_TAR             := tk$(TK_VERSION)-src.tar.gz
READLINE_TAR       := readline-$(READLINE_VERSION).tar.gz
LIBUUID_TAR        := libuuid-$(LIBUUID_VERSION).tar.gz
OMPI_TAR           := openmpi-$(OMPI_VERSION).tar.gz
PETSC_TAR          := petsc-$(PETSC_VERSION).tar.gz
PRECICE_TAR        := v$(PRECICE_VERSION).tar.gz
EIGEN_TAR          := eigen-$(EIGEN_VERSION).tar.gz
BOOST_TAR          := boost_1_68_0.tar.gz
LIBXML2_TAR        := libxml2_v$(LIBXML2_VERSION).tar.gz

# Directory configurations
include .env

VENV_DIR    := ${PWD}/.venv
SOURCES_DIR := ${PWD}/.sources
BUILD_DIR   := $(SOURCES_DIR)/build

# Build parameters
NPROC     := $(shell nproc)
DOWNLOAD  := wget -nc

# Compiler flags
export PATH            := ${VENV_DIR}/bin:${VENV_DIR}/sbin:/bin:/usr/bin
export LIBDIR          := $(VENV_DIR)/lib:$(VENV_DIR)/lib64
export PKG_CONFIG_PATH := $(VENV_DIR)/lib/pkgconfig:$(VENV_DIR)/lib64/pkgconfig
export CPPFLAGS        := -I$(VENV_DIR)/include \
                          -I$(VENV_DIR)/include/ncurses \
						  -I$(VENV_DIR)/include/openssl \
						  -I$(VENV_DIR)/include/boost \
						  -I$(VENV_DIR)/include/petsc \
						  -I$(VENV_DIR)/include/openmpi \
						  -I$(VENV_DIR)/include/python3.12
export LDFLAGS         := -L$(VENV_DIR)/lib -L$(VENV_DIR)/lib/hwloc -L$(VENV_DIR)/lib64 -L$(VENV_DIR)/lib/pmix -L$(VENV_DIR)/lib/openmpi -Wl,-rpath,$(VENV_DIR)/lib$(VENV_DIR)/lib/hwloc:$(VENV_DIR)/lib64:$(VENV_DIR)/lib/pmix:$(VENV_DIR)/lib/openmpi
export CFLAGS          := -O2
export CXXFLAGS        := -O2

export BOOST_ROOT      := $(VENV_DIR)
export PETSC_ARCH      := arch-linux
export Eigen3_ROOT     := $(VENV_DIR)/include/eigen

#-------------------------------------------------------------------------------
# Main Targets
#-------------------------------------------------------------------------------
.PHONY: all clean

.PHONY: pip

pip:
	pip $(filter-out $@,$(MAKECMDGOALS))
	pip freeze > requirements.lock

%:
	@:

python: $(VENV_DIR)/.python.done
	@echo "Build completed successfully"

python_env: $(VENV_DIR)/.venv.done
	@echo "Build completed successfully"

petsc: $(VENV_DIR)/.petsc.done
	@echo "Build completed successfully"

precice: $(VENV_DIR)/.precice.done
	@echo "Build completed successfully"

all: petsc python python_env precice  
	@echo "Build completed successfully"

clean:
	rm -rf $(VENV_DIR) $(SOURCES_DIR)/build

#-------------------------------------------------------------------------------
# Dependency Management
#-------------------------------------------------------------------------------

# Download all sources
$(SOURCES_DIR)/download.done:
	@mkdir -p $(SOURCES_DIR)
	@echo "Downloading source packages..."
	@cd $(SOURCES_DIR) && $(DOWNLOAD) \
		https://archives.boost.io/release/$(BOOST_VERSION)/source/$(BOOST_TAR) \
		https://download.open-mpi.org/release/open-mpi/v4.1/$(OMPI_TAR) \
		https://ftp.gnu.org/gnu/gdbm/$(GDBM_TAR) \
		https://ftp.gnu.org/gnu/ncurses/$(NCURSES_TAR) \
		https://ftp.gnu.org/gnu/readline/$(READLINE_TAR) \
		https://github.com/libffi/libffi/releases/download/v$(LIBFFI_VERSION)/$(LIBFFI_TAR) \
		https://github.com/precice/precice/archive/$(PRECICE_TAR) \
		https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/$(EIGEN_TAR) \
		https://gitlab.gnome.org/GNOME/libxml2/-/archive/v$(LIBXML2_VERSION)/$(LIBXML2_TAR) \
		https://prdownloads.sourceforge.net/tcl/$(TCL_TAR) \
		https://prdownloads.sourceforge.net/tcl/$(TK_TAR) \
		https://sourceforge.net/projects/libuuid/files/$(LIBUUID_TAR) \
		https://sourceware.org/pub/bzip2/$(BZIP2_TAR) \
		https://web.cels.anl.gov/projects/petsc/download/release-snapshots/$(PETSC_TAR) \
		https://www.openssl.org/source/$(OPENSSL_TAR) \
		https://www.python.org/ftp/python/$(PYTHON_VERSION)/$(PYTHON_TAR) \
		https://www.sqlite.org/2025/$(SQLITE_TAR) \
		https://bitbucket.org/petsc/pkg-fblaslapack/get/v3.4.2-p3.tar.gz \
		https://bitbucket.org/petsc/pkg-metis/get/v5.1.0-p12.tar.gz \
		https://bitbucket.org/petsc/pkg-parmetis/get/v4.0.3-p9.tar.gz \
		https://github.com/hypre-space/hypre/archive/ee74c20e7a84e4e48eec142c6bb6ff2a75db72f1.tar.gz \
		https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz \
		https://github.com/Reference-ScaLAPACK/scalapack/archive/0234af94c6578c53ac4c19f2925eb6e5c4ad6f0f.tar.gz \
		https://github.com/xiaoyeli/superlu_dist/archive/eac44cf48878f8699cc19fb566832b6736596727.tar.gz \
		https://gitlab.com/slepc/slepc/-/archive/v3.22.0/slepc-v3.22.0.tar.gz \
		https://gitlab.inria.fr/scotch/scotch/-/archive/v7.0.5/scotch-v7.0.5.tar.gz \
		https://mumps-solver.org/MUMPS_5.7.3.tar.gz \
		https://web.cels.anl.gov/projects/petsc/download/externalpackages/f2cblaslapack-3.8.0.q2.tar.gz \
		https://web.cels.anl.gov/projects/petsc/download/externalpackages/hdf5-1.14.3-p1.tar.bz2
	@touch $@

#-------------------------------------------------------------------------------
# Dependency Build Rules
#-------------------------------------------------------------------------------

# Pattern rule for building libraries
define build-library
$(VENV_DIR)/.$(1).done: $(SOURCES_DIR)/download.done
	@echo "Building $(1)..."
	@mkdir -p $(BUILD_DIR)/$(1)
	@tar -xzf $(SOURCES_DIR)/$(2) -C $(BUILD_DIR)/$(1) --strip-components=1
	@cd $(BUILD_DIR)/$(1) && \
		./configure --prefix=$(VENV_DIR) $(3) && \
		make -j$(NPROC) && \
		make install
	@touch $$@
endef

# Build definitions using the pattern rule
$(eval $(call build-library,libffi,$(LIBFFI_TAR),--disable-static))
$(eval $(call build-library,sqlite,$(SQLITE_TAR),))
$(eval $(call build-library,ncurses,$(NCURSES_TAR),--with-shared --with-termlib --without-debug))
$(eval $(call build-library,gdbm,$(GDBM_TAR),))
$(eval $(call build-library,libuuid,$(LIBUUID_TAR),))
$(eval $(call build-library,openmpi,$(OMPI_TAR),--disable-java))

# Special case for openssl (no configure script)
$(VENV_DIR)/.openssl.done: $(SOURCES_DIR)/download.done
	@echo "Compilando OpenSSL..."
	@mkdir -p $(BUILD_DIR)/openssl
	@tar -xzf $(SOURCES_DIR)/$(OPENSSL_TAR) -C $(BUILD_DIR)/openssl --strip-components=1
	@cd $(BUILD_DIR)/openssl && \
		./config --prefix=$(VENV_DIR) --openssldir=/etc/ssl && \
		make -j1 depend && \
		make install_sw
	@touch $@

# Special case for bzip2 (no configure script)
$(VENV_DIR)/.bzip2.done: $(SOURCES_DIR)/download.done
	@echo "Building bzip2..."
	@mkdir -p $(BUILD_DIR)/bzip2
	@tar -xzf $(SOURCES_DIR)/$(BZIP2_TAR) -C $(BUILD_DIR)/bzip2 --strip-components=1
	@cd $(BUILD_DIR)/bzip2 && \
		make CFLAGS="-Wall -Winline -O2 -g -fPIC" LDFLAGS="-fPIC" && \
		make install PREFIX=$(VENV_DIR)
	@touch $@

# Special case for Tcl/Tk
$(VENV_DIR)/.tcl.done: $(SOURCES_DIR)/download.done
	@echo "Building Tcl..."
	@mkdir -p $(BUILD_DIR)/tcl
	@tar -xzf $(SOURCES_DIR)/$(TCL_TAR) -C $(BUILD_DIR)/tcl --strip-components=1
	@cd $(BUILD_DIR)/tcl/unix && \
		./configure --prefix=$(VENV_DIR) && \
		make -j$(NPROC) && \
		make install
	@touch $@

$(VENV_DIR)/.tk.done: $(VENV_DIR)/.tcl.done
	@echo "Building Tk..."
	@mkdir -p $(BUILD_DIR)/tk
	@tar -xzf $(SOURCES_DIR)/$(TK_TAR) -C $(BUILD_DIR)/tk --strip-components=1
	@cd $(BUILD_DIR)/tk/unix && \
		./configure --prefix=$(VENV_DIR) && \
		make -j$(NPROC) && \
		make install
	@touch $@

# Special case for Readline
$(VENV_DIR)/.readline.done: $(SOURCES_DIR)/download.done $(VENV_DIR)/.ncurses.done
	@echo "Building Readline..."
	@mkdir -p $(BUILD_DIR)/readline
	@tar -xzf $(SOURCES_DIR)/$(READLINE_TAR) -C $(BUILD_DIR)/readline --strip-components=1
	@cd $(BUILD_DIR)/readline && \
		./configure LDFLAGS="-L$(VENV_DIR)/lib -lncurses" --prefix=$(VENV_DIR) --with-curses --with-shared-termcap-library  --with-shared --with-termlib && \
		make -j$(NPROC) && \
		make install
	@touch $@

#-------------------------------------------------------------------------------
# Python Build
#-------------------------------------------------------------------------------

# $(VENV_DIR)/.bzip2.done
# $(VENV_DIR)/.sqlite.done
$(VENV_DIR)/.python.done: $(VENV_DIR)/.libffi.done \
	$(VENV_DIR)/.openssl.done \
	$(VENV_DIR)/.ncurses.done \
	$(VENV_DIR)/.gdbm.done \
	$(VENV_DIR)/.tcl.done \
	$(VENV_DIR)/.tk.done \
	$(VENV_DIR)/.readline.done 
	@echo "Building Python $(PYTHON_VERSION)..."
	@mkdir -p $(BUILD_DIR)/python
	@tar -xzf $(SOURCES_DIR)/$(PYTHON_TAR) -C $(BUILD_DIR)/python --strip-components=1
	@cd $(BUILD_DIR)/python && \
		./configure \
			--prefix=$(VENV_DIR) \
			--enable-optimizations \
			--with-tcltk-includes="-I$(VENV_DIR)/include" \
			--with-tcltk-libs="-L$(VENV_DIR)/lib -ltcl8.6 -ltk8.6" && \
		make -j$(NPROC) && \
		make install
	@ln -sf $(VENV_DIR)/bin/python3 $(VENV_DIR)/bin/python
	@touch $@

#-------------------------------------------------------------------------------
# Virtual Environment Setup
#-------------------------------------------------------------------------------

$(VENV_DIR)/.venv.done: $(VENV_DIR)/.python.done $(VENV_DIR)/.petsc.done
	@echo "Setting up virtual environment..."
	@$(VENV_DIR)/bin/pip3 install --upgrade pip setuptools wheel
	@$(VENV_DIR)/bin/pip3 install -r requirements.txt
	@$(VENV_DIR)/bin/pip3 install $(BUILD_DIR)/petsc/src/binding/petsc4py
	@ln -sf $(VENV_DIR)/bin/pip3 $(VENV_DIR)/bin/pip
	@touch $@

#------------------------------------------------------------------------------
# Instalar PETSc y dependencias
#------------------------------------------------------------------------------
$(VENV_DIR)/.petsc.done: $(SOURCES_DIR)/download.done $(VENV_DIR)/.openmpi.done
	@echo "Instalando PETSc..."
	@mkdir -p $(BUILD_DIR)/petsc
	@tar -xzf $(SOURCES_DIR)/$(PETSC_TAR) -C $(BUILD_DIR)/petsc --strip-components=1
	@cd $(BUILD_DIR)/petsc && \
		./configure \
			LDFLAGS=$$LDFLAGS \
			--prefix=$(VENV_DIR) \
			--with-packages-download-dir=$(SOURCES_DIR) \
			--with-shared-libraries=1 \
			--with-debugging=0 \
			--download-fblaslapack \
			--download-f2cblaslapack \
			--download-hdf5 \
			--download-hypre \
			--download-metis \
			--download-mumps \
			--download-parmetis \
			--download-ptscotch \
			--download-slepc \
			--download-scalapack \
			--download-superlu_dist \
			--download-zlib && \
		make -j$(NPROC) PETSC_DIR=$(BUILD_DIR)/petsc PETSC_ARCH=arch-linux all && \
		make PETSC_DIR=$(BUILD_DIR)/petsc PETSC_ARCH=arch-linux install
		make install
	@touch $@


#------------------------------------------------------------------------------
# Instalar dependencias y preCICE
#------------------------------------------------------------------------------
$(VENV_DIR)/.eigen.done:
	@echo "Instalando Eigen..."
	@tar -xzf $(SOURCES_DIR)/$(EIGEN_TAR) -C $(VENV_DIR)/include/
	@mv $(VENV_DIR)/include/eigen-$(EIGEN_VERSION) $(VENV_DIR)/include/eigen
	@touch $@

$(VENV_DIR)/.boost.done:
	@echo "Instalando Boost..."
	@mkdir -p $(BUILD_DIR)/boost
	@tar -xzf $(SOURCES_DIR)/$(BOOST_TAR) -C $(BUILD_DIR)/boost --strip-components=1
	@cd $(BUILD_DIR)/boost && \
		./bootstrap.sh --prefix=$(VENV_DIR) --with-libraries=log,thread,system,filesystem,program_options,test && \
		./b2 --prefix=$(VENV_DIR) install 
	@touch $@

$(VENV_DIR)/.libxml2.done:
	@echo "Instalando libxml2..."
	@mkdir -p $(BUILD_DIR)/libxml2
	@tar -xf $(SOURCES_DIR)/$(LIBXML2_TAR) -C $(BUILD_DIR)/libxml2 --strip-components=1
	@cd $(BUILD_DIR)/libxml2 && \
		./autogen.sh && \
		./configure --prefix=$(VENV_DIR) --without-python && \
		make -j$(NPROC) && make install
	@touch $@

$(VENV_DIR)/.precice.done: $(VENV_DIR)/.petsc.done \
	$(VENV_DIR)/.libxml2.done \
	$(VENV_DIR)/.eigen.done \
	$(VENV_DIR)/.boost.done
	@echo "Instalando preCICE..."
	@mkdir -p $(BUILD_DIR)/precice
	@tar -xf $(SOURCES_DIR)/$(PRECICE_TAR) -C $(BUILD_DIR)/precice --strip-components=1
	@cd $(BUILD_DIR)/precice && \
		cmake --preset=production \
			-DCMAKE_BUILD_TYPE=Release \
			-DPYTHON_EXECUTABLE=$(VENV_DIR)/bin/python \
			-DCMAKE_INSTALL_PREFIX=$(VENV_DIR) \
			-DPRECICE_InstallTest=ON \
			-DPRECICE_CONFIGURE_PACKAGE_GENERATION=OFF \
			-DMPI_C_COMPILER=$(VENV_DIR)/bin/mpicc \
			-DMPI_CXX_COMPILER=$(VENV_DIR)/bin/mpicxx \
			-DEIGEN3_INCLUDE_DIR=$(VENV_DIR)/include/eigen
		cd $(BUILD_DIR)/precice/build  && \
				make -j$(NPROC) && \
				make install
	@$(VENV_DIR)/bin/pip3 install pyprecice==$(PRECICE_VERSION)
	@touch $@
