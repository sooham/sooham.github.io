.PHONY: all install clean generate build deploy server watch

# Default target: build for production
all: build

# Install dependencies and theme if needed
install: 
	@echo "Installing npm dependencies..."
	npm ci
	@echo "Ensuring theme is present..."
	@if [ ! -d "themes/polk" ]; then \
		echo "Cloning Polk theme..."; \
		git clone --depth=1 https://github.com/sooham/polk themes/polk; \
	else \
		echo "Theme already present."; \
	fi

# Clean generated files
clean:
	@echo "Cleaning public/ and cache..."
	npm run hexo clean

# Generate static site for production
generate: clean install
	@echo "Generating static site..."
	npm run hexo generate -- $(ARGS)


# Deploy to GitHub Pages (or configured deploy target)
deploy: generate
	@echo "Deploying site..."
	npm run hexo deploy

# Start local development server (does not build for production)
server: install
	@echo "Starting development server..."
	npm run hexo server

# Start server with file watching (for development)
watch: install
	@echo "Starting server with watch mode..."
	npm run hexo server -- --watch

