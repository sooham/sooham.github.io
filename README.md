# Computer Stuff and Other Stuff
## Sooham's Blog

** Setup ** 
```bash
npm install
npm update
npm audit fix --force
npm outdated # should be empty
```
**_config.yml**
Hexo blog settings

**Scaffolds**
YML templates used by default. Edit these to edit generated templates.

**Source Folder**
Source folder. This is where you put your site’s content. Hexo ignores hidden files and files or folders whose names are prefixed with _ (underscore) - except the \_posts folder. Renderable files (e.g. Markdown, HTML) will be processed and put into the public folder, while other files will simply be copied.

**hexo-renderer-kramed**
The npm dependency hexo-renderer-kramed has replaced hexo-renderer-marked.

**commands**
`hexo` can be accessed by `npm run hexo -- <flags>`

|     Create new blog post    |            **hexo new [layout] title**           |
|:---------------------------:|:------------------------------------------------:|
|    Generate static files    | **hexo generate [--deploy] [--force] [--watch]** |
| Start localhost:4000 server |           **hexo server [--port=port]**          |
|        Publish draft        |        **hexo publish [layout] filename**        |
|            Clean            |                  **hexo clean**                  |
|        Deploy to Github     |                 **hexo deploy**                  |


