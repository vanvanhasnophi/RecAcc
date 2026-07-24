# Chronicle Design Language — Lite

深色模式 · 固定主题色 · 零配置 · 适合快速搭建

---

## 设计原则

- **纯深色**：无亮色切换，无媒体查询，无 `data-theme`
- **CSS 变量驱动**：换项目只需改 `:root` 里的值
- **毛玻璃 (Glassmorphism)**：`backdrop-filter: blur()` + 半透明背景
- **可变字体优先**：Inter Variable，字重用 `font-variation-settings`

---

## 1. Token 定义（直接复制）

```css
:root {
  /* === 字体 === */
  --font-sans: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 
    'InterVariable', 'Inter', 'Segoe UI', 'Roboto',
    'PingFang SC', 'Microsoft YaHei', sans-serif;
  --font-serif: 'Noto Serif SC', serif;
  --font-mono: 'Consolas', 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;

  color-scheme: dark;
  font-family: var(--font-sans);
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;

  /* === 品牌 === */
  --accent: #2ea35f;
  --accent-dark: #24804a;
  --accent-contrast: #fff;

  /* === 语义 === */
  --featured: #ffd700;
  --warning: #ff4d50;
  --error: #d9534f;
  --success: #5cb85c;

  /* === 背景 === */
  --bg-primary: #141414;
  --bg-secondary: #1e1e1e;
  --surface: #222222;
  --surface-alt: #2a2a2a;
  --surface-blur: rgba(34, 34, 34, 0.9);
  --surface-blur-alt: rgba(72, 72, 72, 0.5);
  --surface-hover: rgba(255, 255, 255, 0.08);
  --surface-active: rgba(255, 255, 255, 0.14);

  /* === 文字 === */
  --text-primary: #e0e0e0;
  --text-secondary: #a9a9a9;

  /* === 边框 === */
  --border: rgba(169, 169, 169, 0.2);
  --border-blur: rgba(113, 113, 113, 0.2);

  /* === 阴影 === */
  --shadow-1: 0 4px 12px rgba(0, 0, 0, 0.3);
  --shadow-2: 0 4px 12px rgba(0, 0, 0, 0.25);
  --shadow-3: 0 10px 30px rgba(0, 0, 0, 0.5);

  /* === 代码 === */
  --code-bg: rgba(255, 0, 0, 0.12);
  --code-text: #ff8585;

  /* === 圆角 === */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-full: 50%;
  --radius-pill: 9999px;
}
```

---

## 2. 排版

```css
body {
  margin: 0;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 1rem;
  line-height: 1.5;
}

/* 字重 */
body    { font-variation-settings: 'wght' 400; }
h1-h6   { font-variation-settings: 'wght' 570; font-weight: 600; }
strong  { font-variation-settings: 'wght' 700; font-weight: 700; }
a,button{ font-variation-settings: 'wght' 500; font-weight: 500; }

/* 字号层级 */
h1 { font-size: 2rem; }
h2 { font-size: 1.75rem; }
h3 { font-size: 1.5rem; }
h4 { font-size: 1.3rem; }
h5 { font-size: 1.2rem; }
h6 { font-size: 1.1rem; }

a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-dark); text-decoration: underline; }
```

---

## 3. 基础元素

```css
/* 按钮 */
button, .btn {
  border-radius: var(--radius-md);
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font: inherit;
  font-weight: 500;
  background: var(--surface-alt);
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.2s ease;
}
button:hover, .btn:hover { border-color: var(--accent); }

.btn-primary { background: var(--accent); color: var(--accent-contrast); }
.btn-primary:hover { background: var(--accent-dark); }
.btn-ghost { background: transparent; }

/* 输入框 */
input, textarea, select {
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font: inherit;
  font-size: 0.95rem;
  outline: none;
  box-sizing: border-box;
}
input:focus, textarea:focus, select:focus {
  border-color: var(--accent);
  box-shadow: var(--shadow-2);
}

/* 滚动条 */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(165, 165, 165, 0.5);
  border-radius: var(--radius-pill);
}
```

---

## 4. 毛玻璃面板

最核心的视觉特征，一行类名即可：

```css
.glass {
  background: var(--surface-blur);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-blur);
  border-radius: var(--radius-lg);
}
```

用于导航栏、侧边栏、卡片、弹出面板。

---

## 5. 富文本容器

包裹所有渲染后的 markdown/富文本内容：

```css
.rich-content {
  color: var(--text-primary);
  line-height: 1.7;
  font-size: 1rem;
}

.rich-content h1-h6 { margin: 1rem 0 1.2rem; line-height: 1.3; }

.rich-content p { margin: 0 0 1.2rem; }
.rich-content ul, .rich-content ol { margin: 0 0 1.2rem; padding-left: 2rem; }
.rich-content li { margin: 0.6rem 0; }
.rich-content li::marker { color: rgba(224, 224, 224, 0.5); }

.rich-content blockquote {
  border-left: 4px solid var(--accent);
  background: var(--surface-active);
  padding: 0.5em 1em;
  margin: 0.5em 0;
  border-radius: var(--radius-sm);
}

.rich-content table {
  width: 100%; border-collapse: collapse;
  margin: 1.8rem 0;
  border: 1px solid var(--border);
  border-radius: 0.5rem; overflow: hidden;
}
.rich-content th {
  background: color-mix(in srgb, #808080 50%, var(--surface-blur));
  font-weight: 600; padding: 8px 12px; text-align: left;
}
.rich-content td {
  background: var(--surface-blur-alt);
  padding: 8px 12px;
}

.rich-content hr {
  border: none; border-top: 1px solid var(--border);
  margin: 1.5em 0;
}

/* 行内代码 */
.rich-content :not(pre) > code {
  background: var(--code-bg); color: var(--code-text);
  border-radius: 5px; padding: 0.2em 0.5em;
  font-size: 0.85em; font-family: var(--font-mono);
}
```

---

## 6. 图片渐进加载

```css
.img-wrapper {
  position: relative;
  overflow: hidden;
  border-radius: var(--radius-md);
  background: var(--surface-blur-alt);
}

/* 未加载时保持 2:1 占位 */
.img-wrapper:not(:has(.img.loaded)):not([data-error]) {
  aspect-ratio: 2 / 1;
}

.img-wrapper::before {
  content: 'Loading...';
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-secondary);
  font-size: 14px;
}

.img-wrapper[data-error]::before {
  content: 'Image not found';
  color: var(--error);
}

.img-wrapper .img {
  position: relative; z-index: 1;
  width: 100%; display: block;
  border-radius: var(--radius-md);
  opacity: 0; filter: blur(10px);
  transition: opacity 0.3s ease, filter 0.3s ease-out;
}

.img-wrapper .img.loaded {
  opacity: 1; filter: blur(0);
}
```

---

## 7. 代码块

```css
.code-block {
  display: flex; flex-direction: column;
  background: var(--surface-blur-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin: 8px 0;
  font-family: var(--font-mono);
}

.code-block-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.2rem 1rem;
  font-family: var(--font-sans);
  font-size: 0.92rem;
}

.code-block-body {
  overflow: auto;
  padding: 0.7rem 1.5rem 1.2rem;
  font-size: 13.5px;
  line-height: 1.3;
  tab-size: 2;
}

.code-block-footer {
  padding: 0.1rem 1rem 0.2rem;
  font-size: 0.75rem;
  color: var(--text-secondary);
  font-family: var(--font-sans);
}
```

---

## 8. 动效

```css
/* 过渡时长 */
--dur-fast: 0.15s;
--dur-normal: 0.2s;
--dur-slow: 0.3s;

/* 淡入 */
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }

/* 上浮出现 */
@keyframes float-up {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* 悬停微交互 */
.card:hover, .btn:hover { transform: translateY(-1px); }
.card:active, .btn:active { transform: translateY(0); }
```

---

## 9. 响应式

两个断点足够覆盖绝大多数场景：

```css
@media (max-width: 768px) {
  html { --mobile: 1; }  /* 信号量, 供 JS 或 descendant 选择器判断 */
  h1 { font-size: 1.5rem; }
  .glass { backdrop-filter: none; }   /* 移动端降级 */
}

@media (max-width: 480px) {
  h1 { font-size: 1.3rem; }
}
```

移动端注意：
- hover 效果用 `@media (hover: hover)` 包裹
- 触屏用 `:active` 替代 `:hover` 做反馈
- 使用 `100dvh` + `env(safe-area-inset-*)` 适配 iOS

---

## 快速启动

1. 复制 `:root { ... }` 到项目全局 CSS
2. 引入 Inter 可变字体（或回退系统字体）
3. 按需复制对应组件 CSS
4. 覆盖 `--accent` 换品牌色

完整版见 `design-language.md`（含亮色主题、语义化 token 体系、动效 token、完整富文本样式规范）。