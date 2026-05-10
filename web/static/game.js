(function () {
  const STORAGE_SOLO = "race2048-game-id";

  const page = document.getElementById("app-page");
  const modeSelect = document.getElementById("mode-select");
  const sectionSolo = document.getElementById("section-solo");
  const sectionVersus = document.getElementById("section-versus");
  const sectionArena = document.getElementById("section-arena");

  const toastSolo = document.getElementById("toast-solo");
  const toastVersus = document.getElementById("toast-versus");
  const toastArena = document.getElementById("toast-arena");

  /** @type {string | null} */
  let soloGameId = localStorage.getItem(STORAGE_SOLO);
  /** @type {{ matchId: string, playerId: string, botId: string } | null} */
  let duel = null;
  /** @type {{ matchId: string, leftId: string, rightId: string } | null} */
  let arena = null;

  /** Arena: auto-play both bots until a win or both stuck. */
  let arenaAutoActive = false;

  let soloToastTimer = 0;

  function makeTarget(gridId, wrapId) {
    return {
      grid: /** @type {HTMLDivElement} */ (document.getElementById(gridId)),
      wrap: /** @type {HTMLDivElement} */ (document.getElementById(wrapId)),
      /** @type {number[][] | null} */
      lastBoard: null,
    };
  }

  const soloT = makeTarget("grid-solo", "board-wrap-solo");
  const versusYou = makeTarget("grid-you", "board-wrap-you");
  const versusBot = makeTarget("grid-bot", "board-wrap-bot");
  const arenaL = makeTarget("grid-arena-l", "board-wrap-arena-l");
  const arenaR = makeTarget("grid-arena-r", "board-wrap-arena-r");

  function cloneBoard(b) {
    return b.map((row) => row.slice());
  }

  function tileClass(n) {
    if (!n) return "cell--empty";
    if (n <= 2048) return "cell--n" + n;
    let p = 4096;
    while (p < n && p < 1000000) p *= 2;
    if (n >= 4096) return "cell--n4096";
    return "cell--n2048";
  }

  /**
   * @param {{ grid: HTMLDivElement, wrap: HTMLElement, lastBoard: number[][] | null }} t
   * @param {{ animate?: boolean }} opts
   */
  function renderInto(t, board, opts = {}) {
    const animate = opts.animate !== false && typeof gsap !== "undefined";
    const prevBoard = t.lastBoard;
    t.grid.innerHTML = "";
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const v = board[r][c];
        const div = document.createElement("div");
        div.className = "cell " + tileClass(v);
        const inner = document.createElement("span");
        inner.className = "cell-inner";
        inner.textContent = v ? String(v) : "";
        div.appendChild(inner);
        t.grid.appendChild(div);
      }
    }

    t.lastBoard = cloneBoard(board);

    if (!animate || !prevBoard) return;

    requestAnimationFrame(() => {
      const cells = t.grid.querySelectorAll(".cell");
      for (let i = 0; i < 16; i++) {
        const r = Math.floor(i / 4);
        const c = i % 4;
        const pv = prevBoard[r][c];
        const nv = board[r][c];
        const cell = cells[i];

        if (!nv && !pv) continue;

        if (!pv && nv) {
          gsap.from(cell, {
            scale: 0.5,
            opacity: 0.2,
            duration: 0.2,
            ease: "back.out(1.6)",
          });
          continue;
        }

        if (pv > 0 && nv > pv && nv === pv * 2) {
          gsap.fromTo(
            cell,
            { scale: 1.14 },
            { scale: 1, duration: 0.18, ease: "power3.out" }
          );
        }
      }
    });
  }

  function shakeWrap(wrapEl) {
    if (!wrapEl) return;
    if (typeof gsap !== "undefined") {
      gsap.killTweensOf(wrapEl);
      const tl = gsap.timeline({ defaults: { duration: 0.042, ease: "none" } });
      tl.to(wrapEl, { x: -12 })
        .to(wrapEl, { x: 12 })
        .to(wrapEl, { x: -9 })
        .to(wrapEl, { x: 9 })
        .to(wrapEl, { x: -5 })
        .to(wrapEl, { x: 5 })
        .to(wrapEl, { x: 0, duration: 0.05 });
    } else {
      wrapEl.classList.remove("is-shaking");
      void wrapEl.offsetWidth;
      wrapEl.classList.add("is-shaking");
      window.setTimeout(() => wrapEl.classList.remove("is-shaking"), 380);
    }
  }

  function setSoloToast(msg, ms = 1800) {
    toastSolo.textContent = msg;
    window.clearTimeout(soloToastTimer);
    soloToastTimer = window.setTimeout(() => {
      toastSolo.textContent = "";
    }, ms);
  }

  function formatStatus(snap) {
    if (snap.won) return "Reached 2048!";
    if (snap.game_over) return "Game over — no moves left.";
    return "Playing (" + snap.legal_actions.length + " legal moves)";
  }

  async function loadCheckpointList() {
    try {
      const res = await fetch("/api/checkpoints");
      if (!res.ok) return;
      const data = await res.json();
      const dl = document.getElementById("checkpoint-list");
      if (!dl) return;
      dl.innerHTML = "";
      for (const name of data.checkpoints || []) {
        const o = document.createElement("option");
        o.value = name;
        dl.appendChild(o);
      }
    } catch {
      /* offline */
    }
  }

  function applyModeUI() {
    const m = modeSelect.value;
    sectionSolo.classList.toggle("hidden", m !== "solo");
    sectionVersus.classList.toggle("hidden", m !== "versus");
    sectionArena.classList.toggle("hidden", m !== "arena");
    page.classList.toggle("page--wide", m === "versus" || m === "arena");
  }

  function arenaAutoResetBtn() {
    const btn = document.getElementById("arena-auto");
    if (btn) btn.textContent = "Auto until 2048";
  }

  function stopArenaAuto() {
    arenaAutoActive = false;
    arenaAutoResetBtn();
  }

  modeSelect.addEventListener("change", () => {
    stopArenaAuto();
    applyModeUI();
  });
  applyModeUI();

  /* ---------- Solo ---------- */
  const statusSolo = document.getElementById("status-solo");
  const seedInput = document.getElementById("seed");

  async function soloNewGame() {
    const raw = seedInput.value.trim();
    const qs =
      raw === ""
        ? ""
        : "?seed=" + encodeURIComponent(Number.parseInt(raw, 10));
    if (qs && Number.isNaN(Number.parseInt(raw, 10))) {
      setSoloToast("Seed must be an integer.");
      return;
    }
    const res = await fetch("/api/games" + qs, { method: "POST" });
    if (!res.ok) {
      setSoloToast("Could not start game.");
      return;
    }
    const data = await res.json();
    soloGameId = data.id;
    localStorage.setItem(STORAGE_SOLO, soloGameId);
    soloT.lastBoard = null;
    renderInto(soloT, data.board, { animate: false });
    statusSolo.textContent = formatStatus(data);
    setSoloToast("");
  }

  async function soloStep(action) {
    if (!soloGameId) {
      await soloNewGame();
      if (!soloGameId) return;
    }
    const prev = soloT.lastBoard ? cloneBoard(soloT.lastBoard) : null;
    const res = await fetch("/api/games/" + encodeURIComponent(soloGameId) + "/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (res.status === 404) {
      setSoloToast("Session expired — restarting.");
      soloGameId = null;
      localStorage.removeItem(STORAGE_SOLO);
      await soloNewGame();
      return;
    }
    if (!res.ok) {
      setSoloToast("Move failed.");
      return;
    }
    const data = await res.json();

    if (!data.valid) {
      shakeWrap(soloT.wrap);
      statusSolo.textContent = formatStatus(data);
      return;
    }

    renderInto(soloT, data.board, { animate: true });
    statusSolo.textContent = formatStatus(data);
    setSoloToast("");
  }

  document.getElementById("new-game").addEventListener("click", () => soloNewGame());
  document.querySelectorAll("#dpad-solo .dir").forEach((btn) => {
    btn.addEventListener("click", () => soloStep(btn.getAttribute("data-dir")));
  });

  async function soloBootstrap() {
    if (modeSelect.value !== "solo") return;
    if (soloGameId) {
      const res = await fetch("/api/games/" + encodeURIComponent(soloGameId));
      if (res.ok) {
        const data = await res.json();
        soloT.lastBoard = null;
        renderInto(soloT, data.board, { animate: false });
        statusSolo.textContent = formatStatus(data);
        return;
      }
      soloGameId = null;
      localStorage.removeItem(STORAGE_SOLO);
    }
    await soloNewGame();
  }

  /* ---------- Versus ---------- */
  const statusYou = document.getElementById("status-versus-you");
  const statusBot = document.getElementById("status-versus-bot");
  const versusOppSel = document.getElementById("versus-opponent");
  const versusCkpt = document.getElementById("versus-checkpoint");
  const versusSeed = document.getElementById("versus-seed");
  document.getElementById("versus-start").addEventListener("click", startVersus);
  document.getElementById("versus-step-bot-only").addEventListener("click", () => opponentTurn());

  async function fetchSnap(gameId) {
    const r = await fetch("/api/games/" + encodeURIComponent(gameId));
    if (!r.ok) throw new Error("bad game");
    return r.json();
  }

  async function startVersus() {
    toastVersus.textContent = "";
    const opponent = versusOppSel.value;
    const ck = versusCkpt.value.trim() || null;
    const raw = versusSeed.value.trim();
    const body = { mode: "versus", opponent, checkpoint: ck };
    if (raw !== "") {
      const s = Number.parseInt(raw, 10);
      if (Number.isNaN(s)) {
        toastVersus.textContent = "Seed must be an integer.";
        return;
      }
      body.seed = s;
    }
    const res = await fetch("/api/matches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toastVersus.textContent = err.detail || "Could not start duel (DQN needs checkpoints/*.pt).";
      return;
    }
    const d = await res.json();
    duel = { matchId: d.match_id, playerId: d.left_game_id, botId: d.right_game_id };
    document.getElementById("versus-bot-title").textContent =
      opponent === "dqn" ? "DQN bot" : opponent + " bot";

    versusYou.lastBoard = null;
    versusBot.lastBoard = null;
    const [sy, sb] = await Promise.all([fetchSnap(duel.playerId), fetchSnap(duel.botId)]);
    renderInto(versusYou, sy.board, { animate: false });
    renderInto(versusBot, sb.board, { animate: false });
    statusYou.textContent = formatStatus(sy);
    statusBot.textContent = formatStatus(sb);
  }

  async function versusPlayerStep(action) {
    if (!duel) {
      toastVersus.textContent = "Start a duel first.";
      return;
    }
    const prev = versusYou.lastBoard ? cloneBoard(versusYou.lastBoard) : null;
    const res = await fetch("/api/games/" + encodeURIComponent(duel.playerId) + "/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!res.ok) {
      toastVersus.textContent = "Move failed.";
      return;
    }
    const data = await res.json();
    if (!data.valid) {
      shakeWrap(versusYou.wrap);
      statusYou.textContent = formatStatus(data);
      return;
    }
    renderInto(versusYou, data.board, { animate: true });
    statusYou.textContent = formatStatus(data);
    await opponentTurn();
  }

  async function opponentTurn() {
    if (!duel) return;
    const prevB = versusBot.lastBoard ? cloneBoard(versusBot.lastBoard) : null;
    const res = await fetch(
      "/api/matches/" + encodeURIComponent(duel.matchId) + "/opponent-turn",
      { method: "POST" }
    );
    if (!res.ok) {
      toastVersus.textContent = "Bot step failed.";
      return;
    }
    const data = await res.json();
    renderInto(versusBot, data.board, { animate: true });
    statusBot.textContent = formatStatus(data);
    if (!data.valid && prevB) {
      /* bot board unchanged (e.g. already terminal) */
    }
  }

  document.querySelectorAll("#dpad-you .dir").forEach((btn) => {
    btn.addEventListener("click", () => versusPlayerStep(btn.getAttribute("data-dir")));
  });

  /* ---------- Arena ---------- */
  const policies = [
    ["random", "Random"],
    ["ordered", "Ordered"],
    ["greedy", "Greedy"],
    ["dqn", "DQN"],
  ];
  const arenaLeft = document.getElementById("arena-left");
  const arenaRight = document.getElementById("arena-right");
  for (const [v, l] of policies) {
    arenaLeft.appendChild(new Option(l, v));
    arenaRight.appendChild(new Option(l, v));
  }
  arenaRight.selectedIndex = 1;

  document.getElementById("arena-start").addEventListener("click", startArena);
  document.getElementById("arena-tick-left").addEventListener("click", () => arenaTick(["left"]));
  document.getElementById("arena-tick-right").addEventListener("click", () => arenaTick(["right"]));
  document.getElementById("arena-tick-both").addEventListener("click", () =>
    arenaTick(["left", "right"])
  );
  document.getElementById("arena-auto").addEventListener("click", toggleArenaAuto);

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * @param {any} pack
   * @param {{ animate?: boolean }} opts
   */
  function applyArenaSteps(pack, opts = {}) {
    const animate = opts.animate !== false;
    const steps = pack.steps || {};
    if (steps.left) {
      renderInto(arenaL, steps.left.board, { animate });
      document.getElementById("status-arena-left").textContent = formatStatus(steps.left);
    }
    if (steps.right) {
      renderInto(arenaR, steps.right.board, { animate });
      document.getElementById("status-arena-right").textContent = formatStatus(steps.right);
    }
  }

  async function arenaTickPayload(lanes) {
    if (!arena) return null;
    const res = await fetch(
      "/api/matches/" + encodeURIComponent(arena.matchId) + "/tick",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lanes }),
      }
    );
    if (!res.ok) return null;
    return res.json();
  }

  async function startArena() {
    stopArenaAuto();
    toastArena.textContent = "";
    const left = arenaLeft.value;
    const right = arenaRight.value;
    const body = {
      mode: "arena",
      left,
      right,
      left_checkpoint: document.getElementById("arena-left-ckpt").value.trim() || null,
      right_checkpoint: document.getElementById("arena-right-ckpt").value.trim() || null,
    };
    const raw = document.getElementById("arena-seed").value.trim();
    if (raw !== "") {
      const s = Number.parseInt(raw, 10);
      if (Number.isNaN(s)) {
        toastArena.textContent = "Seed must be an integer.";
        return;
      }
      body.seed = s;
    }
    const res = await fetch("/api/matches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toastArena.textContent = err.detail || "Arena start failed.";
      return;
    }
    const d = await res.json();
    arena = { matchId: d.match_id, leftId: d.left_game_id, rightId: d.right_game_id };
    document.getElementById("arena-left-title").textContent =
      arenaLeft.options[arenaLeft.selectedIndex].text;
    document.getElementById("arena-right-title").textContent =
      arenaRight.options[arenaRight.selectedIndex].text;

    arenaL.lastBoard = null;
    arenaR.lastBoard = null;
    const [sl, sr] = await Promise.all([fetchSnap(arena.leftId), fetchSnap(arena.rightId)]);
    renderInto(arenaL, sl.board, { animate: false });
    renderInto(arenaR, sr.board, { animate: false });
    document.getElementById("status-arena-left").textContent = formatStatus(sl);
    document.getElementById("status-arena-right").textContent = formatStatus(sr);
  }

  async function arenaTick(lanes) {
    const pack = await arenaTickPayload(lanes);
    if (!pack) {
      toastArena.textContent = arena ? "Tick failed." : "Start arena first.";
      return;
    }
    applyArenaSteps(pack, { animate: true });
  }

  const ARENA_AUTO_MAX_ITER = 100_000;
  const ARENA_AUTO_DELAY_MS = 32;

  async function toggleArenaAuto() {
    if (!arena) {
      toastArena.textContent = "Start arena first.";
      return;
    }
    if (arenaAutoActive) {
      stopArenaAuto();
      return;
    }

    arenaAutoActive = true;
    const btn = document.getElementById("arena-auto");
    btn.textContent = "Stop";

    toastArena.textContent = "";

    try {
      for (let iter = 0; iter < ARENA_AUTO_MAX_ITER && arenaAutoActive && arena; iter++) {
        const pack = await arenaTickPayload(["left", "right"]);
        if (!pack) {
          toastArena.textContent = "Tick failed.";
          break;
        }
        applyArenaSteps(pack, { animate: false });

        const steps = pack.steps || {};
        const sl = steps.left;
        const sr = steps.right;
        if (!sl || !sr) {
          toastArena.textContent = "Unexpected server response.";
          break;
        }

        if (sl.won || sr.won) {
          if (sl.won && sr.won) {
            toastArena.textContent = "Both reached 2048 this step.";
          } else if (sl.won) {
            toastArena.textContent = "Left wins — reached 2048.";
          } else {
            toastArena.textContent = "Right wins — reached 2048.";
          }
          break;
        }
        if (sl.game_over && sr.game_over) {
          toastArena.textContent =
            "Stopped — neither bot reached 2048; both boards are stuck.";
          break;
        }

        await sleep(ARENA_AUTO_DELAY_MS);
      }
      if (
        arenaAutoActive &&
        toastArena.textContent === "" &&
        arena
      ) {
        toastArena.textContent =
          "Stopped — safety step limit reached (reload or adjust code if needed).";
      }
    } finally {
      stopArenaAuto();
    }
  }

  /* ---------- Global keys ---------- */
  function onKey(e) {
    const map = {
      ArrowUp: "up",
      ArrowDown: "down",
      ArrowLeft: "left",
      ArrowRight: "right",
    };
    const a = map[e.key];
    if (!a) return;
    const t = e.target;
    if (t instanceof HTMLElement && t.closest("input, textarea, select")) return;
    e.preventDefault();
    const m = modeSelect.value;
    if (m === "solo") soloStep(a);
    else if (m === "versus") versusPlayerStep(a);
  }

  document.addEventListener("keydown", onKey, { passive: false });

  loadCheckpointList();
  soloBootstrap();
})();
