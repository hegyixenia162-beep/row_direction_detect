"use strict";

const canvas = document.getElementById("annotationCanvas");
const ctx = canvas.getContext("2d");
const viewport = document.getElementById("canvasViewport");
const state = {
  images: [], currentIndex: -1, image: null, rows: [], predictions: [], selectedIndex: -1,
  tool: "draw", drawing: false, panning: false, draft: null, dirty: false,
  scale: 1, offsetX: 0, offsetY: 0, panStart: null, history: [],
};

// 把服务端请求封装为统一函数；非 2xx 响应会携带中文错误并中断当前操作。
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  // 只有响应成功才返回数据，为什么：避免保存失败后界面错误地显示“已保存”。
  if (!response.ok) throw new Error(payload.error || `请求失败：${response.status}`);
  return payload;
}

// 短暂显示非阻塞提示，用户可以继续画线而无需关闭弹窗。
function toast(message) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("show"), 1800);
}

// 更新保存状态；dirty 表示当前画布与磁盘 JSON 不一致。
function setSaveState(mode) {
  state.dirty = mode === "dirty";
  const element = document.getElementById("saveState");
  element.dataset.state = mode;
  element.querySelector("span").textContent = mode === "dirty" ? "未保存" : mode === "saving" ? "保存中" : "已保存";
}

// 读取图像队列并刷新左侧完成进度，不改变当前图时尽量保持用户位置。
async function loadImageList() {
  const currentName = state.currentIndex >= 0 ? state.images[state.currentIndex]?.name : null;
  state.images = (await api("/api/images")).images;
  renderImageList();
  const annotated = state.images.filter((item) => item.annotated).length;
  document.getElementById("progressText").textContent = `${annotated} / ${state.images.length}`;
  document.getElementById("progressBar").style.width = `${state.images.length ? 100 * annotated / state.images.length : 0}%`;
  // 首次加载打开第一张；刷新列表时继续停留在同一文件。
  if (state.images.length && state.currentIndex < 0) await openImage(0);
  else if (currentName) state.currentIndex = state.images.findIndex((item) => item.name === currentName);
}

// 渲染影像队列；已保存图像用绿色圆点和行数标记，便于快速查漏。
function renderImageList() {
  const list = document.getElementById("imageList");
  list.replaceChildren();
  // 每个队列项绑定索引，点击后先自动保存当前图再切换。
  state.images.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = `image-item ${item.annotated ? "done" : ""} ${index === state.currentIndex ? "active" : ""}`;
    button.innerHTML = `<i></i><span title="${item.name}">${item.name}</span><em>${item.annotated ? item.row_count : "—"}</em>`;
    button.addEventListener("click", () => openImage(index));
    list.appendChild(button);
  });
}

// 打开指定图像，并并行加载人工标注与只读算法参考层。
async function openImage(index) {
  // 索引越界时不切图，防止快捷键在首尾产生异常请求。
  if (index < 0 || index >= state.images.length || index === state.currentIndex) return;
  // 切图前保存未提交修改，避免连续标注时因忘记点击保存而丢数据。
  if (state.dirty) await saveCurrent(false);
  state.currentIndex = index;
  state.selectedIndex = -1;
  state.history = [];
  renderImageList();
  const item = state.images[index];
  document.getElementById("emptyState").classList.remove("hidden");
  try {
    const [annotation, prediction] = await Promise.all([
      api(`/api/annotation/${encodeURIComponent(item.name)}`),
      api(`/api/prediction/${encodeURIComponent(item.name)}`),
    ]);
    state.rows = Array.isArray(annotation.rows) ? annotation.rows.map(normalizeRow) : [];
    state.predictions = Array.isArray(prediction.rows) ? prediction.rows.map(normalizeRow) : [];
    const image = new Image();
    image.onload = () => {
      state.image = image;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      fitToViewport();
      document.getElementById("emptyState").classList.add("hidden");
      document.getElementById("imageInfo").textContent = `${item.name} · ${image.naturalWidth} × ${image.naturalHeight}px`;
      setSaveState("saved");
      render();
    };
    image.onerror = () => toast("原图加载失败");
    image.src = `/api/image/${encodeURIComponent(item.name)}`;
  } catch (error) {
    toast(error.message);
  }
}

// 把任意输入行规范为原图坐标数值，保留 row_id 供后续评估追踪。
function normalizeRow(row, index = 0) {
  return { row_id: row.row_id ?? index + 1, start: row.start.map(Number), end: row.end.map(Number) };
}

// 根据视口计算完整显示比例；留出边距是为了让靠边端点仍易于点击。
function fitToViewport() {
  // 图像尚未加载时不执行计算，防止自然尺寸为零。
  if (!state.image) return;
  const padding = 24;
  state.scale = Math.min((viewport.clientWidth - padding * 2) / canvas.width, (viewport.clientHeight - padding * 2) / canvas.height);
  state.offsetX = (viewport.clientWidth - canvas.width * state.scale) / 2;
  state.offsetY = (viewport.clientHeight - canvas.height * state.scale) / 2;
  updateTransform();
}

// 把画布 CSS 变换同步到当前缩放和平移状态；内部像素仍保持原图分辨率。
function updateTransform() {
  canvas.style.transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.scale})`;
  document.getElementById("zoomBadge").textContent = `${Math.round(state.scale * 100)}%`;
}

// 把鼠标视口坐标转换为原图像素坐标，这是标注可用于定量比对的关键。
function eventToImage(event) {
  const rect = viewport.getBoundingClientRect();
  const x = (event.clientX - rect.left - state.offsetX) / state.scale;
  const y = (event.clientY - rect.top - state.offsetY) / state.scale;
  return [Math.max(0, Math.min(canvas.width - 1, x)), Math.max(0, Math.min(canvas.height - 1, y))];
}

// 深复制当前行数组并压入撤销栈，避免后续对象修改同时污染历史状态。
function pushHistory() {
  state.history.push(JSON.stringify(state.rows));
  // 只保留最近 100 步，为什么：足够人工回退，同时避免长时间标注无限占用内存。
  if (state.history.length > 100) state.history.shift();
}

// 计算点到有限线段的距离，选择工具用它判断鼠标点中了哪一行。
function pointSegmentDistance(point, start, end) {
  const vx = end[0] - start[0], vy = end[1] - start[1];
  const wx = point[0] - start[0], wy = point[1] - start[1];
  const lengthSquared = vx * vx + vy * vy;
  // 退化为点的零长度线段直接计算点距，避免除以零。
  if (lengthSquared === 0) return Math.hypot(wx, wy);
  const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / lengthSquared));
  return Math.hypot(point[0] - (start[0] + t * vx), point[1] - (start[1] + t * vy));
}

// 在当前缩放下寻找点击距离最近的人工线段；容差换算到原图坐标后保持屏幕手感一致。
function findRowAt(point) {
  let best = -1, bestDistance = 12 / state.scale;
  // 遍历所有人工线，距离更近且在容差内的候选成为当前选择。
  state.rows.forEach((row, index) => {
    const distance = pointSegmentDistance(point, row.start, row.end);
    if (distance < bestDistance) { best = index; bestDistance = distance; }
  });
  return best;
}

// 绘制一组线段；颜色和虚线区分人工真值、选中线、草稿及算法参考。
function drawRows(rows, color, width, dashed = false) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width / state.scale;
  ctx.setLineDash(dashed ? [12 / state.scale, 8 / state.scale] : []);
  // 每条线显式绘制两端圆点，确保满足“有限线段而非无限直线”的标注语义。
  rows.forEach((row) => {
    ctx.beginPath(); ctx.moveTo(...row.start); ctx.lineTo(...row.end); ctx.stroke();
    ctx.beginPath(); ctx.arc(...row.start, 6 / state.scale, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(...row.end, 6 / state.scale, 0, Math.PI * 2); ctx.fill();
  });
  ctx.restore();
}

// 重绘原图和所有覆盖层；算法参考层默认关闭以减少标注诱导偏差。
function render() {
  // 原图未就绪时没有可绘制内容。
  if (!state.image) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0);
  // 仅在用户主动打开参考开关时显示算法结果，并以青色虚线与人工橙线区分。
  if (document.getElementById("predictionToggle").checked) drawRows(state.predictions, "#42d7e8", 3, true);
  state.rows.forEach((row, index) => drawRows([row], index === state.selectedIndex ? "#fff26b" : "#ff9d24", index === state.selectedIndex ? 6 : 4));
  // 拖画过程中显示半透明草稿，鼠标松开且长度合格后才写入正式数组。
  if (state.draft) drawRows([state.draft], "rgba(255,157,36,.65)", 4);
  document.getElementById("lineInfo").textContent = `人工线段 ${state.rows.length}`;
}

// 保存当前图人工标注；silent 用于切图自动保存，避免频繁弹出成功提示。
async function saveCurrent(showToast = true) {
  // 没有当前图或没有修改时无需发请求；手动点击保存仍给出反馈。
  if (state.currentIndex < 0 || !state.dirty) {
    if (showToast) toast("当前标注已保存");
    return;
  }
  setSaveState("saving");
  const item = state.images[state.currentIndex];
  const payload = {
    schema_version: 1, source: item.name,
    image_size: { width: canvas.width, height: canvas.height },
    coordinate_system: "image_pixels_top_left_origin",
    rows: state.rows.map((row, index) => ({ row_id: index + 1, start: row.start.map(Math.round), end: row.end.map(Math.round) })),
  };
  try {
    await api(`/api/annotation/${encodeURIComponent(item.name)}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    setSaveState("saved");
    item.annotated = true; item.row_count = state.rows.length;
    renderImageList();
    if (showToast) toast(`已保存 ${state.rows.length} 条线段`);
  } catch (error) {
    setSaveState("dirty");
    toast(`保存失败：${error.message}`);
    throw error;
  }
}

// 删除当前选中线并记录撤销状态；没有选中对象时保持画布不变。
function deleteSelected() {
  if (state.selectedIndex < 0) return;
  pushHistory();
  state.rows.splice(state.selectedIndex, 1);
  state.selectedIndex = -1;
  setSaveState("dirty"); render();
}

// 恢复最近一次修改前的完整线段数组。
function undo() {
  // 撤销栈为空时给出轻提示，避免用户误以为快捷键失效。
  if (!state.history.length) { toast("没有可撤销的操作"); return; }
  state.rows = JSON.parse(state.history.pop());
  state.selectedIndex = -1;
  setSaveState("dirty"); render();
}

// 切换绘制/选择模式，并同步工具按钮和鼠标形状。
function setTool(tool) {
  state.tool = tool;
  document.getElementById("drawButton").classList.toggle("active", tool === "draw");
  document.getElementById("selectButton").classList.toggle("active", tool === "select");
  canvas.style.cursor = tool === "draw" ? "crosshair" : "default";
}

// 围绕指定视口点缩放，保证鼠标下的原图位置缩放前后不跳动。
function zoomAt(factor, clientX = viewport.getBoundingClientRect().left + viewport.clientWidth / 2, clientY = viewport.getBoundingClientRect().top + viewport.clientHeight / 2) {
  if (!state.image) return;
  const rect = viewport.getBoundingClientRect();
  const imageX = (clientX - rect.left - state.offsetX) / state.scale;
  const imageY = (clientY - rect.top - state.offsetY) / state.scale;
  const next = Math.max(0.05, Math.min(4, state.scale * factor));
  state.offsetX = clientX - rect.left - imageX * next;
  state.offsetY = clientY - rect.top - imageY * next;
  state.scale = next; updateTransform(); render();
}

// 鼠标按下：右键或空格进入平移，画线模式创建草稿，选择模式命中已有线。
canvas.addEventListener("pointerdown", (event) => {
  if (!state.image) return;
  // 右键或按住空格时平移视图，不修改任何标注。
  if (event.button === 2 || state.spacePressed) {
    state.panning = true; state.panStart = [event.clientX, event.clientY, state.offsetX, state.offsetY];
    canvas.setPointerCapture(event.pointerId); return;
  }
  const point = eventToImage(event);
  // 绘制模式记录起点并创建动态终点；选择模式只更新选中索引。
  if (state.tool === "draw") {
    state.drawing = true; state.draft = { start: point, end: point };
    canvas.setPointerCapture(event.pointerId);
  } else {
    state.selectedIndex = findRowAt(point); render();
  }
});

// 鼠标移动：根据当前交互状态更新画面平移或草稿终点。
canvas.addEventListener("pointermove", (event) => {
  if (state.panning) {
    state.offsetX = state.panStart[2] + event.clientX - state.panStart[0];
    state.offsetY = state.panStart[3] + event.clientY - state.panStart[1]; updateTransform(); return;
  }
  if (state.drawing && state.draft) { state.draft.end = eventToImage(event); render(); }
});

// 鼠标松开：长度足够才提交正式线段，短误触直接丢弃。
canvas.addEventListener("pointerup", (event) => {
  if (state.panning) { state.panning = false; return; }
  if (!state.drawing || !state.draft) return;
  state.draft.end = eventToImage(event);
  const length = Math.hypot(state.draft.end[0] - state.draft.start[0], state.draft.end[1] - state.draft.start[1]);
  // 至少 10 个原图像素才认为是有效拖画，避免单击产生零长度线。
  if (length >= 10) {
    pushHistory(); state.rows.push(normalizeRow(state.draft, state.rows.length));
    state.selectedIndex = state.rows.length - 1; setSaveState("dirty");
  }
  state.drawing = false; state.draft = null; render();
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());
viewport.addEventListener("wheel", (event) => { event.preventDefault(); zoomAt(event.deltaY < 0 ? 1.15 : 1 / 1.15, event.clientX, event.clientY); }, { passive: false });

// 全局快捷键提高连续标注效率，并避免在浏览器里触发默认保存页面行为。
window.addEventListener("keydown", (event) => {
  if (event.code === "Space") { state.spacePressed = true; event.preventDefault(); }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); saveCurrent(); }
  else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); undo(); }
  else if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); deleteSelected(); }
  else if (event.key.toLowerCase() === "v") setTool("draw");
  else if (event.key.toLowerCase() === "s" && !event.ctrlKey) setTool("select");
  else if (event.key.toLowerCase() === "f") fitToViewport();
  else if (event.key.toLowerCase() === "a") openImage(state.currentIndex - 1);
  else if (event.key.toLowerCase() === "d") openImage(state.currentIndex + 1);
});
window.addEventListener("keyup", (event) => { if (event.code === "Space") state.spacePressed = false; });
window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("resize", fitToViewport);

document.getElementById("prevButton").onclick = () => openImage(state.currentIndex - 1);
document.getElementById("nextButton").onclick = () => openImage(state.currentIndex + 1);
document.getElementById("drawButton").onclick = () => setTool("draw");
document.getElementById("selectButton").onclick = () => setTool("select");
document.getElementById("undoButton").onclick = undo;
document.getElementById("deleteButton").onclick = deleteSelected;
document.getElementById("saveButton").onclick = () => saveCurrent();
document.getElementById("fitButton").onclick = fitToViewport;
document.getElementById("zoomInButton").onclick = () => zoomAt(1.2);
document.getElementById("zoomOutButton").onclick = () => zoomAt(1 / 1.2);
document.getElementById("predictionToggle").onchange = render;
document.getElementById("refreshButton").onclick = loadImageList;

loadImageList().catch((error) => toast(`初始化失败：${error.message}`));
