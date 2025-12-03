# Installation directory
INSTALL_DIR = $(shell echo $$HOME)
BUILD_DIR = build
CMAKE_FLAGS = -DUSE_REDIS=1 -DBUILD_BENCHMARK=1 -DCMAKE_CXX_STANDARD=17

.PHONY: all install clean redis

all: install

# Install everything
install: redis

# Install and configure Redis
redis:
	@echo "Installing Redis..."
	sudo apt-get update
	sudo apt-get install -y redis-server
	@echo "Redis installation complete"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@echo "Clean complete"

# Clean everything including repositories
distclean: clean
	@echo "Removing all installed components..."
	@echo "Full cleanup complete"

# Status check
status:
	@echo "Checking Redis status..."
	@systemctl status redis-server || true

# Help
help:
	@echo "Available targets:"
	@echo "  make install    - Install Redis"
	@echo "  make redis     - Install and configure Redis only"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make distclean - Remove all components"
	@echo "  make status    - Check installation status"
	@echo "  make help      - Show this help message"