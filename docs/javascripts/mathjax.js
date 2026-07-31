/* ============================================================
   MathJax 配置
   - 与 mkdocs.yml 中 pymdownx.arithmatex (generic: true) 配合
   - 支持 \mathbb, \boldsymbol, \bm 等常用宏
   ============================================================ */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    macros: {
      // 常用向量/矩阵加粗宏（机器人学教程高频使用）
      bm: ["\\boldsymbol{#1}", 1],
      R: ["\\mathrm{R}"],          // 旋转矩阵命名空间预留
      T: ["\\mathrm{T}"]          // 齐次变换命名空间预留
    }
  },
  options: {
    // 忽略 arithmatex 之外的内容，提升渲染性能
    ignoreHtmlClass: "no-mathjax",
    processHtmlClass: "arithmatex"
  }
};
