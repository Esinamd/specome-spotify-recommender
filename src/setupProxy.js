const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/Songs2',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      methods: ['GET', 'POST'],
    })
  );
  app.use(
    '/Artists2',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      methods: ['GET', 'POST'],
    })
  );
  app.use(
    '/SpotifyConnect',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      methods: ['GET', 'POST'],
    })
  );
};
