all: server

server: generate
	hexo server
generate: clean
	hexo generate

clean: # remove all files in /public/
	hexo clean
