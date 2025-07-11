# Computer Stuff and Other Stuff
## Sooham's Blog

**Setup** 
```bash
npm install
npm update
npm audit fix --force
npm outdated # should be empty
 npm install --save-dev npm-check-updates # check for updates and save
git clone https://github.com/sooham/polk themes/polk
```

## What is Hexo?

**Hexo** is a fast, simple, and powerful static site generator built with Node.js. It's perfect for creating blogs, documentation sites, and personal websites.

### Key Features:
- **⚡ Fast**: Generates hundreds of files in seconds
- **📝 Markdown Support**: Write content in Markdown with full syntax highlighting
- **🎨 Theme System**: Easy to customize with themes and plugins
- **🔧 Plugin System**: Extensible with hundreds of plugins
- **📱 Mobile-Friendly**: Responsive design out of the box
- **🚀 Deployment Ready**: One-command deployment to GitHub Pages, Netlify, Vercel, etc.
- **🌍 Internationalization**: Multi-language support
- **📊 SEO Optimized**: Built-in SEO features and meta tags

### What Hexo Can Do:
- **Blog Creation**: Create and manage blog posts with categories, tags, and archives
- **Static Site Generation**: Build complete static websites from Markdown files
- **Content Management**: Organize content with drafts, pages, and custom layouts
- **Asset Management**: Handle images, CSS, JavaScript, and other assets
- **Template Customization**: Create custom themes and layouts using EJS, Pug, or Handlebars
- **Plugin Integration**: Add features like comments, analytics, search, and more
- **Multi-format Output**: Generate RSS feeds, sitemaps, and JSON files
- **Development Server**: Live preview with hot reload during development

**Official Documentation**: https://hexo.io/docs/

## Project Structure

### Key Files and Directories:

**`_config.yml`**
- Main configuration file for your Hexo site
- Controls site title, URL, theme, plugins, and deployment settings
- Edit this to customize your blog's appearance and behavior

**`scaffolds/`**
- Template files for new posts, pages, and drafts
- Contains `post.md`, `page.md`, and `draft.md` templates
- Edit these to customize the default structure of new content

**`source/`**
- Where all your content lives
- `_posts/`: Published blog posts (Markdown files)
- `_drafts/`: Unpublished posts (won't be generated)
- Other files/folders: Pages, images, and assets
- Hexo processes Markdown/HTML files and copies others directly

**`themes/polk/`**
- Your custom theme directory
- Contains templates, styles, and theme-specific configurations
- Customize the look and feel of your site here

### Content Organization:
- **Posts**: Go in `source/_posts/` with YAML front matter
- **Pages**: Go in `source/` (e.g., `source/about.md`)
- **Drafts**: Go in `source/_drafts/` for unpublished content
- **Assets**: Images, CSS, JS files go in `source/` and are copied to the output

**hexo-renderer-marked**
The npm dependency hexo-renderer-marked has replaced hexo-renderer-kramed.

## Your Hexo Setup

### Current Configuration:
- **Hexo Version**: 7.3.0 (latest)
- **Theme**: Custom Polk theme
- **Renderer**: hexo-renderer-marked (secure)
- **Deployment**: GitHub Pages via hexo-deployer-git

### Available Commands:
`hexo` can be accessed by `npm run hexo -- <flags>`

| Command | Description | Usage |
|:--------|:------------|:------|
| **Create new blog post** | Create a new post with default template | `hexo new [layout] title` |
| **Generate static files** | Build the site for production | `hexo generate [--deploy] [--force] [--watch]` |
| **Start development server** | Preview site locally with live reload | `hexo server [--port=port]` |
| **Publish draft** | Convert draft to published post | `hexo publish [layout] filename` |
| **Clean cache** | Remove generated files and cache | `hexo clean` |
| **Deploy to GitHub** | Build and deploy to GitHub Pages | `hexo deploy` |

### Common Workflows:

**Create and Preview a New Post:**
```bash
hexo new "My New Blog Post"
hexo server
# Edit the post in source/_posts/
# View at http://localhost:4000
```

**Build and Deploy:**
```bash
hexo clean
hexo generate
hexo deploy
```

**Quick Development:**
```bash
hexo server --watch
# This starts the server and watches for file changes
```


