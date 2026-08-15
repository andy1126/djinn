import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  // F13:纯逻辑单测(纯函数,node 环境即可,不引入 jsdom)
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        // F8:手动分包,避免 echarts/editor/antd 全进首屏单 chunk
        manualChunks: {
          echarts: ['echarts', 'echarts-for-react'],
          editor: ['@uiw/react-codemirror', '@codemirror/lang-python'],
          antd: ['antd', '@ant-design/icons'],
          vendor: [
            'react',
            'react-dom',
            'react-router-dom',
            'axios',
            'zustand',
            '@tanstack/react-query',
            'dayjs',
          ],
        },
      },
    },
  },
})