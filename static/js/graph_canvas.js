/**
 * Interactive HTML5 Canvas Bipartite Graph Visualizer
 * Simulates force-directed relaxation, node dragging, zoom/pan, and hover highlighting.
 */

class BipartiteGraphViewer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.nodes = [];
    this.links = [];
    this.nodeMap = new Map();

    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    this.isDragging = false;
    this.draggedNode = null;
    this.isPanning = false;
    this.lastMousePos = { x: 0, y: 0 };
    this.hoveredNode = null;

    this.simulationRunning = false;
    this.animFrameId = null;

    this.initCanvasSize();
    this.bindEvents();
  }

  initCanvasSize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width || 800;
    this.height = rect.height || 480;

    // Support High DPI Displays
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);

    this.panX = this.width / 2;
    this.panY = this.height / 2;
  }

  loadGraph(graphData) {
    if (!graphData || !graphData.nodes) return;

    this.nodes = [];
    this.links = [];
    this.nodeMap.clear();

    const targetUser = graphData.target_user;

    // Initialize node coordinates in structured bipartite / radial layout
    graphData.nodes.forEach((n, idx) => {
      let initialX = 0;
      let initialY = 0;

      if (n.type === 'target_user') {
        initialX = -200;
        initialY = 0;
      } else if (n.type === 'interacted_item') {
        const angle = (idx / 6) * Math.PI - Math.PI / 2;
        initialX = -60 + Math.cos(angle) * 80;
        initialY = Math.sin(angle) * 140;
      } else if (n.type === 'similar_user') {
        const angle = (idx / 6) * Math.PI - Math.PI / 2;
        initialX = 80 + Math.cos(angle) * 90;
        initialY = Math.sin(angle) * 150;
      } else if (n.type === 'recommended_item') {
        const offset = (idx - 2) * 55;
        initialX = 220;
        initialY = offset;
      }

      const nodeObj = {
        ...n,
        x: initialX + (Math.random() - 0.5) * 20,
        y: initialY + (Math.random() - 0.5) * 20,
        vx: 0,
        vy: 0,
        radius: n.size || 14
      };

      this.nodes.push(nodeObj);
      this.nodeMap.set(n.id, nodeObj);
    });

    graphData.links.forEach(link => {
      const source = this.nodeMap.get(link.source);
      const target = this.nodeMap.get(link.target);
      if (source && target) {
        this.links.push({
          source,
          target,
          weight: link.weight || 1.0,
          type: link.type || 'normal',
          dashed: link.dashed || false
        });
      }
    });

    this.startSimulation();
  }

  startSimulation() {
    this.simulationRunning = true;
    this.alpha = 1.0;

    const tick = () => {
      this.updatePhysics();
      this.render();

      if (this.alpha > 0.005) {
        this.alpha *= 0.96;
        this.animFrameId = requestAnimationFrame(tick);
      } else {
        this.simulationRunning = false;
        this.render();
      }
    };

    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    this.animFrameId = requestAnimationFrame(tick);
  }

  updatePhysics() {
    const kRepel = 2400;
    const kSpring = 0.05;
    const springLen = 110;

    // Node Repulsion
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        const a = this.nodes[i];
        const b = this.nodes[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const distSq = dx * dx + dy * dy + 1;
        const dist = Math.sqrt(distSq);

        if (dist < 350) {
          const force = (kRepel / distSq) * this.alpha;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          if (a !== this.draggedNode) { a.vx -= fx; a.vy -= fy; }
          if (b !== this.draggedNode) { b.vx += fx; b.vy += fy; }
        }
      }
    }

    // Link Spring Attraction
    for (const link of this.links) {
      const a = link.source;
      const b = link.target;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const displacement = dist - springLen;
      const force = displacement * kSpring * this.alpha;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;

      if (a !== this.draggedNode) { a.vx += fx; a.vy += fy; }
      if (b !== this.draggedNode) { b.vx -= fx; b.vy -= fy; }
    }

    // Apply velocities with damping
    for (const node of this.nodes) {
      if (node === this.draggedNode) continue;
      node.vx *= 0.85;
      node.vy *= 0.85;
      node.x += node.vx;
      node.y += node.vy;
    }
  }

  render() {
    this.ctx.save();
    this.ctx.clearRect(0, 0, this.width, this.height);

    // Apply viewport transform (pan & zoom)
    this.ctx.translate(this.panX, this.panY);
    this.ctx.scale(this.zoom, this.zoom);

    // 1. Draw Links
    for (const link of this.links) {
      this.ctx.beginPath();
      this.ctx.moveTo(link.source.x, link.source.y);
      this.ctx.lineTo(link.target.x, link.target.y);

      if (link.dashed) {
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeStyle = 'rgba(236, 72, 153, 0.65)';
        this.ctx.lineWidth = 2.2;
      } else if (link.type === 'consumed') {
        this.ctx.setLineDash([]);
        this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
        this.ctx.lineWidth = 1.6;
      } else if (link.type === 'shared') {
        this.ctx.setLineDash([]);
        this.ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
        this.ctx.lineWidth = 1.4;
      } else {
        this.ctx.setLineDash([]);
        this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
        this.ctx.lineWidth = 1.2;
      }

      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    // 2. Draw Nodes
    for (const node of this.nodes) {
      const isHovered = this.hoveredNode === node;
      const r = node.radius + (isHovered ? 4 : 0);

      // Outer glow for hovered or special nodes
      if (isHovered || node.type === 'target_user' || node.type === 'recommended_item') {
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, r + 5, 0, Math.PI * 2);
        this.ctx.fillStyle = node.color ? `${node.color}33` : 'rgba(99, 102, 241, 0.25)';
        this.ctx.fill();
      }

      // Main Node Circle
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      this.ctx.fillStyle = node.color || '#6366f1';
      this.ctx.fill();
      this.ctx.strokeStyle = '#ffffff';
      this.ctx.lineWidth = isHovered ? 2.5 : 1.5;
      this.ctx.stroke();

      // Node Label
      this.ctx.font = `${isHovered ? 'bold ' : ''}11px Inter, sans-serif`;
      this.ctx.fillStyle = '#f8fafc';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'top';

      // Truncate label if too long
      const labelText = node.label.length > 20 ? node.label.substring(0, 18) + '...' : node.label;
      this.ctx.fillText(labelText, node.x, node.y + r + 5);
    }

    this.ctx.restore();
  }

  screenToWorld(sx, sy) {
    return {
      x: (sx - this.panX) / this.zoom,
      y: (sy - this.panY) / this.zoom
    };
  }

  findNodeAt(wx, wy) {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      const dx = wx - n.x;
      const dy = wy - n.y;
      if (dx * dx + dy * dy <= (n.radius + 6) * (n.radius + 6)) {
        return n;
      }
    }
    return null;
  }

  bindEvents() {
    window.addEventListener('resize', () => {
      this.initCanvasSize();
      this.render();
    });

    this.canvas.addEventListener('mousedown', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const world = this.screenToWorld(sx, sy);

      const target = this.findNodeAt(world.x, world.y);
      if (target) {
        this.isDragging = true;
        this.draggedNode = target;
      } else {
        this.isPanning = true;
        this.lastMousePos = { x: sx, y: sy };
      }
    });

    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const world = this.screenToWorld(sx, sy);

      if (this.isDragging && this.draggedNode) {
        this.draggedNode.x = world.x;
        this.draggedNode.y = world.y;
        this.draggedNode.vx = 0;
        this.draggedNode.vy = 0;
        this.alpha = 0.3;
        if (!this.simulationRunning) this.startSimulation();
      } else if (this.isPanning) {
        this.panX += sx - this.lastMousePos.x;
        this.panY += sy - this.lastMousePos.y;
        this.lastMousePos = { x: sx, y: sy };
        this.render();
      } else {
        const found = this.findNodeAt(world.x, world.y);
        if (found !== this.hoveredNode) {
          this.hoveredNode = found;
          this.canvas.style.cursor = found ? 'pointer' : 'grab';
          this.render();
        }
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
      this.draggedNode = null;
      this.isPanning = false;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      this.zoom = Math.min(Math.max(this.zoom * zoomFactor, 0.4), 2.5);
      this.render();
    }, { passive: false });
  }

  resetView() {
    this.zoom = 1.0;
    this.panX = this.width / 2;
    this.panY = this.height / 2;
    this.alpha = 0.5;
    this.startSimulation();
  }

  zoomIn() {
    this.zoom = Math.min(this.zoom * 1.2, 2.5);
    this.render();
  }

  zoomOut() {
    this.zoom = Math.max(this.zoom / 1.2, 0.4);
    this.render();
  }
}

window.BipartiteGraphViewer = BipartiteGraphViewer;
