#!/usr/bin/env node
/**
 * `npm run web` — starts Metro (which serves the web bundle) and a tiny
 * static server for public/index.html. Zero extra dependencies.
 *
 *   Page:   http://localhost:3000
 *   Bundle: http://localhost:8081/index.bundle?platform=web&dev=true
 */
const { spawn } = require('node:child_process');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const PAGE_PORT = process.env.WEB_PORT || 3000;
const root = path.join(__dirname, '..');

// 1. Metro dev server (serves /index.bundle?platform=web).
const metro = spawn('npx', ['react-native', 'start'], {
  cwd: root,
  stdio: 'inherit',
  shell: process.platform === 'win32',
});
process.on('exit', () => metro.kill());
process.on('SIGINT', () => process.exit(0));

// 2. Static server for the shell page.
const server = http.createServer((req, res) => {
  const file = path.join(root, 'public', 'index.html');
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(500);
      res.end('index.html not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(data);
  });
});
server.listen(PAGE_PORT, () => {
  console.log(`\n  AntID web → http://localhost:${PAGE_PORT}\n`);
});
