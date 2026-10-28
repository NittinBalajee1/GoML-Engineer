(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_CSV = 12 * 1024 * 1024;
  let modelReady = false;
  let hasSelectedMessages = false;

  async function readUtf8(file) {
    return new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer());
  }

  function message(element, text) {
    if (!element) return;
    element.textContent = text || '';
    element.hidden = !text;
  }

  async function postJson(path, data) {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    return result;
  }

  function download(filename, content, contentType) {
    const url = URL.createObjectURL(new Blob([content], { type: contentType }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  function bindDrop(zone, input, callback) {
    if (!zone || !input) return;
    for (const name of ['dragenter', 'dragover']) zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add('dragging');
    });
    for (const name of ['dragleave', 'drop']) zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove('dragging');
    });
    zone.addEventListener('drop', (event) => callback(Array.from(event.dataTransfer.files || [])));
    input.addEventListener('change', () => callback(Array.from(input.files || [])));
  }

  function initPlacement() {
    const input = $('placement-file');
    const submit = $('placement-submit');
    if (!input || !submit) return;
    const error = $('placement-error');
    const result = $('placement-result');
    let file = null;
    let predictions = [];

    function accept(files) {
      message(error, '');
      file = files[0] || null;
      if (files.length > 1) {
        file = null;
        message(error, 'Choose one CSV file at a time.');
      } else if (file && !file.name.toLowerCase().endsWith('.csv')) {
        file = null;
        message(error, 'Please choose a CSV file.');
      } else if (file && file.size > MAX_CSV) {
        file = null;
        message(error, 'The CSV exceeds the 12 MB limit.');
      }
      $('placement-file-label').textContent = file ? file.name : 'Choose or drop a CSV file';
      submit.disabled = !file;
    }

    bindDrop($('placement-drop'), input, accept);
    submit.addEventListener('click', async () => {
      if (!file) return;
      message(error, '');
      result.hidden = true;
      submit.disabled = true;
      submit.textContent = 'Scoring…';
      try {
        const data = await postJson('/api/placement', { csv: await readUtf8(file) });
        predictions = data.predictions;
        $('stat-total').textContent = String(data.count);
        $('stat-placed').textContent = String(data.placed_count);
        $('stat-not-placed').textContent = String(data.not_placed_count);
        const body = $('placement-rows');
        body.replaceChildren();
        for (const row of predictions) {
          const tr = document.createElement('tr');
          const id = document.createElement('td');
          const prediction = document.createElement('td');
          id.textContent = row.student_id;
          prediction.textContent = row.placed === 1 ? 'Placed' : 'Not placed';
          tr.append(id, prediction);
          body.append(tr);
        }
        result.hidden = false;
        result.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (cause) {
        message(error, cause.message || 'Could not score this CSV.');
      } finally {
        submit.textContent = 'Score students';
        submit.disabled = !file;
      }
    });

    $('placement-download').addEventListener('click', () => {
      const quote = (value) => `"${String(value).replaceAll('"', '""')}"`;
      const csv = ['student_id,placed', ...predictions.map((row) => `${quote(row.student_id)},${row.placed}`)];
      download('placement_predictions.csv', `${csv.join('\r\n')}\r\n`, 'text/csv;charset=utf-8');
    });
  }

  function initMessages() {
    const input = $('messages-files');
    const submit = $('messages-submit');
    if (!input || !submit) return;
    const error = $('messages-error');
    const result = $('messages-result');
    let files = [];
    let tasks = [];

    function accept(selected) {
      message(error, '');
      $('messages-warning').hidden = true;
      files = selected;
      hasSelectedMessages = files.length > 0;
      const total = files.reduce((sum, file) => sum + file.size, 0);
      if (files.some((file) => !file.name.toLowerCase().endsWith('.txt'))) {
        files = [];
        message(error, 'Only .txt files are accepted.');
      } else if (files.length > 50) {
        files = [];
        message(error, 'Choose no more than 50 files per run.');
      } else if (files.some((file) => file.size > 512 * 1024)) {
        files = [];
        message(error, 'Each message must be 512 KB or smaller.');
      } else if (total > 8 * 1024 * 1024) {
        files = [];
        message(error, 'Combined message files must be 8 MB or smaller.');
      }
      $('messages-file-label').textContent = files.length === 1 ? files[0].name : files.length ? `${files.length} files selected` : 'Choose or drop .txt files';
      hasSelectedMessages = files.length > 0;
      const chips = $('selected-messages');
      chips.replaceChildren();
      for (const file of files) {
        const chip = document.createElement('span');
        chip.className = 'file-chip';
        chip.textContent = file.name;
        chips.append(chip);
      }
      chips.hidden = files.length === 0;
      submit.disabled = !files.length || !modelReady;
    }

    bindDrop($('messages-drop'), input, accept);
    submit.addEventListener('click', async () => {
      if (!files.length || !modelReady) return;
      message(error, '');
      $('messages-warning').hidden = true;
      result.hidden = true;
      submit.disabled = true;
      submit.textContent = 'Running local model…';
      try {
        const messages = await Promise.all(files.map(async (file) => ({ name: file.name, content: await readUtf8(file) })));
        const data = await postJson('/api/messages', { messages });
        tasks = data.tasks;
        let usageRows = Array.isArray(data.message_usage) ? data.message_usage : [];
        let usageSummary = data.usage_summary || {};
        // Older local server instances return extractor stderr as diagnostics.
        // Parse the structured usage lines there so a UI refresh can use the
        // current extractor even before the server process is restarted.
        if (!usageRows.length && typeof data.diagnostics === 'string') {
          const remainingDiagnostics = [];
          for (const line of data.diagnostics.split(/\r?\n/)) {
            if (line.startsWith('message_usage=')) {
              try { usageRows.push(JSON.parse(line.slice('message_usage='.length))); } catch { remainingDiagnostics.push(line); }
            } else if (line.startsWith('usage_summary=')) {
              try { usageSummary = JSON.parse(line.slice('usage_summary='.length)); } catch { remainingDiagnostics.push(line); }
            } else if (!line.startsWith('processed=')) {
              remainingDiagnostics.push(line);
            }
          }
          data.diagnostics = remainingDiagnostics.join('\n');
        }
        $('usage-prompt-total').textContent = Number.isFinite(usageSummary.prompt_tokens) ? usageSummary.prompt_tokens.toLocaleString() : 'Unavailable';
        $('usage-output-total').textContent = Number.isFinite(usageSummary.output_tokens) ? usageSummary.output_tokens.toLocaleString() : 'Unavailable';
        $('usage-time-total').textContent = Number.isFinite(usageSummary.elapsed_seconds) ? `${usageSummary.elapsed_seconds.toFixed(1)}s` : 'Unavailable';
        const usageList = $('usage-rows');
        usageList.replaceChildren();
        const usageById = new Map();
        for (const usage of usageRows) {
          if (usage.message_id) usageById.set(usage.message_id, usage);
          const row = document.createElement('tr');
          const values = [
            usage.message_id || usage.source_file || 'Unknown message',
            usage.outcome || 'unknown',
            Number.isFinite(usage.prompt_tokens) ? usage.prompt_tokens.toLocaleString() : 'Unavailable',
            Number.isFinite(usage.output_tokens) ? usage.output_tokens.toLocaleString() : 'Unavailable',
            Number.isFinite(usage.elapsed_seconds) ? `${usage.elapsed_seconds.toFixed(2)}s` : 'Unavailable',
          ];
          for (const value of values) {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.append(cell);
          }
          usageList.append(row);
        }
        $('task-count').textContent = String(tasks.length);
        const list = $('task-list');
        list.replaceChildren();
        if (!tasks.length) {
          const empty = document.createElement('p');
          empty.className = 'empty-state';
          empty.textContent = 'No actionable requests were found.';
          list.append(empty);
        }
        for (const task of tasks) {
          const card = document.createElement('article');
          card.className = 'task-item';
          const id = document.createElement('span');
          id.className = 'task-id';
          id.textContent = task.message_id;
          const content = document.createElement('div');
          const title = document.createElement('p');
          title.className = 'task-text';
          title.textContent = task.task;
          const meta = document.createElement('div');
          meta.className = 'task-meta';
          const owner = document.createElement('span');
          owner.textContent = `Owner: ${task.owner || 'Unassigned'}`;
          const due = document.createElement('span');
          due.textContent = `Due: ${task.due || 'No deadline'}`;
          meta.append(owner, due);
          const usage = usageById.get(task.message_id);
          const tokens = document.createElement('span');
          tokens.textContent = usage
            ? `Tokens: prompt ${Number.isFinite(usage.prompt_tokens) ? usage.prompt_tokens : 'n/a'} · output ${Number.isFinite(usage.output_tokens) ? usage.output_tokens : 'n/a'}`
            : 'Tokens: unavailable';
          meta.append(tokens);
          content.append(title, meta);
          card.append(id, content);
          list.append(card);
        }
        const warning = $('messages-warning');
        warning.textContent = data.partial ? 'Partial result: review the diagnostics before relying on this list.' : '';
        warning.hidden = !data.partial;
        const diagnostics = $('messages-diagnostics');
        diagnostics.textContent = data.diagnostics || '';
        diagnostics.hidden = !data.partial || !data.diagnostics;
        result.hidden = false;
        result.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (cause) {
        message(error, cause.message || 'Could not extract tasks. Check Ollama and the selected files.');
      } finally {
        submit.textContent = 'Extract tasks';
        submit.disabled = !files.length || !modelReady;
      }
    });

    $('messages-download').addEventListener('click', () => {
      download('tasks.json', `${JSON.stringify({ tasks }, null, 2)}\n`, 'application/json;charset=utf-8');
    });
  }

  async function checkHealth() {
    const pill = $('health-pill');
    const label = $('health-text');
    if (!pill || !label) return;
    try {
      const response = await fetch('/api/health', { cache: 'no-store' });
      const health = await response.json();
      modelReady = health.ollama_running && health.model_available;
      label.textContent = !health.ollama_running ? 'Ollama offline' : !health.model_available ? 'Model missing' : 'Local model ready';
      pill.classList.toggle('offline', !modelReady);
    } catch {
      modelReady = false;
      pill.classList.add('offline');
      label.textContent = 'Local service unavailable';
    }
    const submit = $('messages-submit');
    if (submit) submit.disabled = !modelReady || !hasSelectedMessages;
  }

  initPlacement();
  initMessages();
  checkHealth();
})();
