/**
 * Frontend Application Controller for Parallel Graph Recommendation Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  const state = {
    activeDataset: 'movies',
    currentUser: null,
    graphViewer: null,
    benchmarkChart: null,
    systemCores: 4
  };

  // Safe text setter helper to prevent null property access
  function safeSetText(id, val) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = (val !== undefined && val !== null) ? val : '--';
    }
  }

  // Initialize Canvas Graph Viewer
  state.graphViewer = new BipartiteGraphViewer('bipartiteGraphCanvas');

  // DOM Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const datasetSwitcherBtn = document.getElementById('datasetSwitcherBtn');
  const datasetModal = document.getElementById('datasetModal');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const datasetSelector = document.getElementById('datasetSelector');
  const userSelect = document.getElementById('userSelect');
  const generateRecsBtn = document.getElementById('generateRecsBtn');
  const runBenchmarkBtn = document.getElementById('runBenchmarkBtn');
  const parallelToggle = document.getElementById('parallelToggle');
  const workerSliderContainer = document.getElementById('workerSliderContainer');
  const workerSlider = document.getElementById('workerSlider');
  const workerValueDisplay = document.getElementById('workerValueDisplay');

  // Tab Navigation
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-target');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }

      // If switching to Graph tab or Recommendations, re-render canvas
      if (targetId === 'tab-recommendations' && state.graphViewer) {
        setTimeout(() => {
          state.graphViewer.initCanvasSize();
          state.graphViewer.render();
        }, 100);
      }
    });
  });

  // Worker Slider Updates
  if (workerSlider && workerValueDisplay) {
    workerSlider.addEventListener('input', (e) => {
      workerValueDisplay.textContent = e.target.value;
    });
  }

  // Parallel Execution Toggle
  const modeRadios = document.querySelectorAll('input[name="exec_mode"]');
  modeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      if (workerSliderContainer) {
        if (e.target.value === 'parallel') {
          workerSliderContainer.classList.remove('hidden');
        } else {
          workerSliderContainer.classList.add('hidden');
        }
      }
    });
  });

  // Graph Viewport Controls
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnResetGraph = document.getElementById('btnResetGraph');

  if (btnZoomIn) btnZoomIn.addEventListener('click', () => state.graphViewer.zoomIn());
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => state.graphViewer.zoomOut());
  if (btnResetGraph) btnResetGraph.addEventListener('click', () => state.graphViewer.resetView());

  // API Calls
  async function fetchDatasetInfo() {
    try {
      const res = await fetch('/api/dataset-info');
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // Update UI Header & Stats safely
      safeSetText('currentDatasetLabel', data.dataset_name);
      safeSetText('statTotalUsers', data.num_users ? data.num_users.toLocaleString() : '--');
      safeSetText('statTotalItems', data.num_items ? data.num_items.toLocaleString() : '--');
      safeSetText('statTotalEdges', data.num_edges ? data.num_edges.toLocaleString() : '--');
      safeSetText('statDensity', `${data.density_pct}%`);
      safeSetText('statAvgUserDegree', data.avg_user_degree);
      safeSetText('statAvgItemDegree', data.avg_item_degree);

      state.systemCores = data.cpu_cores_available || 4;
      if (workerSlider) {
        workerSlider.max = state.systemCores;
        workerSlider.value = Math.min(4, state.systemCores);
        if (workerValueDisplay) workerValueDisplay.textContent = workerSlider.value;
      }
    } catch (err) {
      console.error('Failed to load dataset info:', err);
    }
  }

  async function fetchUsersList() {
    try {
      const res = await fetch('/api/users');
      const data = await res.json();
      if (!data.users || data.users.length === 0) return;

      userSelect.innerHTML = '';
      data.users.forEach((u, idx) => {
        const opt = document.createElement('option');
        opt.value = u.user_id;
        opt.textContent = `${u.user_id} (${u.interaction_count} items)`;
        if (idx === 0) opt.selected = true;
        userSelect.appendChild(opt);
      });

      state.currentUser = data.users[0].user_id;
      loadUserProfile(state.currentUser);
    } catch (err) {
      console.error('Failed to load users:', err);
    }
  }

  async function loadUserProfile(userId) {
    if (!userId) return;
    try {
      const res = await fetch(`/api/user-profile/${encodeURIComponent(userId)}`);
      const profile = await res.json();
      if (profile.error) return;

      safeSetText('previewUserInteractions', profile.total_interactions);

      const genreTagsContainer = document.getElementById('previewTopGenres');
      if (genreTagsContainer) {
        genreTagsContainer.innerHTML = '';
        profile.top_categories.forEach(cat => {
          const tag = document.createElement('span');
          tag.className = 'genre-tag';
          tag.textContent = `${cat.name} (${cat.count})`;
          genreTagsContainer.appendChild(tag);
        });
      }
    } catch (err) {
      console.error('Failed to load profile:', err);
    }
  }

  userSelect.addEventListener('change', (e) => {
    state.currentUser = e.target.value;
    loadUserProfile(state.currentUser);
  });

  // Generate Recommendations
  async function generateRecommendations() {
    const userId = userSelect.value;
    if (!userId) return;

    const algorithm = document.querySelector('input[name="rec_algo"]:checked').value;
    const isParallel = document.querySelector('input[name="exec_mode"]:checked').value === 'parallel';
    const numWorkers = isParallel ? parseInt(workerSlider.value, 10) : 1;

    const btnText = generateRecsBtn.querySelector('.btn-label');
    const spinner = generateRecsBtn.querySelector('.loading-spinner');
    btnText.textContent = 'Computing...';
    spinner.classList.remove('hidden');
    generateRecsBtn.disabled = true;

    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          algorithm: algorithm,
          num_workers: numWorkers,
          parallel: isParallel,
          top_k: 5
        })
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // 1. Update Performance Banner safely
      safeSetText('bannerExecTime', `${data.execution_time_ms} ms`);
      safeSetText('bannerWorkers', data.num_workers);
      safeSetText('bannerCandidates', data.candidates_evaluated);
      safeSetText('bannerMode', data.mode ? data.mode.toUpperCase() : 'PARALLEL');

      // 2. Render Top-5 Recommendation Cards
      renderRecommendationCards(data.recommendations);

      // 3. Update Subgraph Visualization
      loadSubgraphVisualization(userId, algorithm);
    } catch (err) {
      alert(`Error generating recommendations: ${err.message}`);
    } finally {
      btnText.textContent = 'Generate Top-5 Recommendations';
      spinner.classList.add('hidden');
      generateRecsBtn.disabled = false;
    }
  }

  function renderRecommendationCards(recs) {
    const container = document.getElementById('recommendationsList');
    container.innerHTML = '';

    if (!recs || recs.length === 0) {
      container.innerHTML = '<div style="color: var(--text-muted); padding: 20px; text-align: center;">No recommendations available for this user.</div>';
      return;
    }

    recs.forEach(rec => {
      const card = document.createElement('div');
      card.className = 'rec-card';

      const meta = rec.metadata || {};
      const cat = meta.genres || meta.category || 'General';
      const scoreLabel = rec.score ? rec.score.toLocaleString() : '0';

      card.innerHTML = `
        <div class="rec-rank-badge">#${rec.rank}</div>
        <div class="rec-info">
          <div class="rec-title-row">
            <h3 class="rec-title">${escapeHtml(rec.title)}</h3>
            <span class="rec-score-pill">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
              Score: ${scoreLabel}
            </span>
          </div>
          <div class="rec-meta-tags">
            <span><strong>Item ID:</strong> ${escapeHtml(rec.item_id)}</span>
            <span>•</span>
            <span>${escapeHtml(cat)}</span>
            ${meta.release_year ? `<span>•</span><span>Year: ${meta.release_year}</span>` : ''}
            ${meta.price ? `<span>•</span><span style="color: var(--accent-emerald); font-weight: 600;">$${meta.price}</span>` : ''}
          </div>
          <p class="rec-desc">${escapeHtml(meta.description || 'No description available.')}</p>
          <div class="rec-explanation-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            <span>${escapeHtml(rec.explanation)}</span>
          </div>
        </div>
      `;
      container.appendChild(card);
    });
  }

  async function loadSubgraphVisualization(userId, algorithm) {
    try {
      const res = await fetch(`/api/graph-data/${encodeURIComponent(userId)}?algorithm=${algorithm}&top_k=5`);
      const graphData = await res.json();
      if (graphData.error) return;

      state.graphViewer.loadGraph(graphData);
    } catch (err) {
      console.error('Failed to load subgraph:', err);
    }
  }

  generateRecsBtn.addEventListener('click', generateRecommendations);

  // Benchmarking Execution
  async function runDistributedBenchmark() {
    const algorithm = document.querySelector('input[name="bench_algo"]:checked').value;
    const batchSize = parseInt(document.getElementById('benchBatchSize').value, 10) || 40;

    const selectedWorkers = [];
    document.querySelectorAll('.bench-worker-check:checked').forEach(cb => {
      selectedWorkers.push(parseInt(cb.value, 10));
    });

    if (selectedWorkers.length === 0) {
      alert('Please select at least one worker process count.');
      return;
    }

    const btnText = runBenchmarkBtn.querySelector('.btn-label');
    const spinner = runBenchmarkBtn.querySelector('.loading-spinner');
    const statusMsg = document.getElementById('benchmarkStatusMsg');

    btnText.textContent = 'Running Benchmark...';
    spinner.classList.remove('hidden');
    runBenchmarkBtn.disabled = true;
    statusMsg.textContent = 'Spawning distributed worker processes and executing candidate scoring...';

    try {
      const res = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algorithm: algorithm,
          user_batch_size: batchSize,
          worker_counts: selectedWorkers,
          top_k: 5
        })
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // 1. Populate System Info safely
      safeSetText('benchSysCores', data.total_system_cores);
      safeSetText('benchSysOS', data.system_info ? data.system_info.os : 'Windows');
      safeSetText('benchSysPython', data.system_info ? data.system_info.python_version : '--');
      safeSetText('benchBaselineTime', `${data.sequential_baseline_sec}s`);

      // 2. Render Results Table
      renderBenchmarkTable(data.benchmark_results);

      // 3. Render Matplotlib Chart
      if (data.chart_base64) {
        const chartImg = document.getElementById('benchmarkMatplotlibImg');
        if (chartImg) chartImg.src = `data:image/png;base64,${data.chart_base64}`;
        const chartBox = document.getElementById('chartContainer');
        if (chartBox) chartBox.classList.remove('hidden');
      }

      if (statusMsg) {
        statusMsg.textContent = `Completed! Evaluated ${data.user_batch_size} users across ${data.benchmark_results.length} worker configurations.`;
      }
    } catch (err) {
      if (statusMsg) statusMsg.textContent = `Benchmark failed: ${err.message}`;
      alert(`Benchmark error: ${err.message}`);
    } finally {
      if (btnText) btnText.textContent = 'Run Distributed Benchmark';
      if (spinner) spinner.classList.add('hidden');
      runBenchmarkBtn.disabled = false;
    }
  }

  function renderBenchmarkTable(records) {
    const tbody = document.getElementById('benchmarkTableBody');
    tbody.innerHTML = '';

    records.forEach(rec => {
      const tr = document.createElement('tr');
      const isSeq = rec.workers === 1;

      tr.innerHTML = `
        <td><strong>${rec.workers} ${rec.workers === 1 ? 'Worker (Seq)' : 'Workers (Par)'}</strong></td>
        <td><span style="text-transform: capitalize; color: ${isSeq ? 'var(--text-muted)' : 'var(--accent-emerald)'}; font-weight: 600;">${rec.mode}</span></td>
        <td><strong>${rec.time_sec} s</strong> (${rec.time_ms} ms)</td>
        <td><span class="badge-speedup">${rec.speedup}x</span></td>
        <td><span class="badge-efficiency">${rec.efficiency_pct}%</span></td>
        <td>${rec.throughput_users_per_sec} users/sec</td>
      `;
      tbody.appendChild(tr);
    });
  }

  runBenchmarkBtn.addEventListener('click', runDistributedBenchmark);

  // Dataset Switcher Modal
  if (datasetSwitcherBtn && datasetModal) {
    datasetSwitcherBtn.addEventListener('click', () => {
      datasetModal.classList.add('show');
    });
  }

  if (closeModalBtn && datasetModal) {
    closeModalBtn.addEventListener('click', () => {
      datasetModal.classList.remove('show');
    });
  }

  // Handle Dataset Switch
  const applyDatasetSwitchBtn = document.getElementById('applyDatasetSwitchBtn');
  if (applyDatasetSwitchBtn) {
    applyDatasetSwitchBtn.addEventListener('click', async () => {
      const selectedKey = datasetSelector.value;
      try {
        const res = await fetch('/api/switch-dataset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_key: selectedKey })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        state.activeDataset = selectedKey;
        datasetModal.classList.remove('show');

        // Reload views
        await fetchDatasetInfo();
        await fetchUsersList();
        generateRecommendations();
      } catch (err) {
        alert(`Failed to switch dataset: ${err.message}`);
      }
    });
  }

  // Handle Custom Dataset Upload
  const uploadForm = document.getElementById('uploadDatasetForm');
  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(uploadForm);
      const submitBtn = uploadForm.querySelector('button[type="submit"]');
      submitBtn.disabled = true;
      submitBtn.textContent = 'Uploading & Parsing...';

      try {
        const res = await fetch('/api/upload-dataset', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        alert(data.message);
        datasetModal.classList.remove('show');
        uploadForm.reset();

        await fetchDatasetInfo();
        await fetchUsersList();
        generateRecommendations();
      } catch (err) {
        alert(`Upload error: ${err.message}`);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Upload and Activate';
      }
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Initial Load
  async function init() {
    await fetchDatasetInfo();
    await fetchUsersList();
    generateRecommendations();
  }

  init();
});
