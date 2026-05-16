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
  /** Latest full-step snapshots for partial ticks (e.g. step left only). */
  /** @type {{ left: any, right: any } | null} */
  let arenaStepCache = null;

  /** Arena: auto-play both bots until a 2048 race winner or both stuck. */
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

  function boardMaxTile(board) {
    let m = 0;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const v = board[r][c];
        if (v > m) m = v;
      }
    }
    return m;
  }

  function boardSum(board) {
    let s = 0;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) s += board[r][c];
    }
    return s;
  }

  /** @param {{ board: number[][], game_over: boolean, won: boolean, legal_actions: string[] }} step */
  function formatArenaStatusOnly(step) {
    if (!step) return "…";
    if (step.won) return "Has 2048 tile";
    if (step.game_over) return "No legal moves";
    return "Playing · " + step.legal_actions.length + " moves";
  }

  function resetArenaScoreTrackerDisplay() {
    for (const id of [
      "arena-score-left-max",
      "arena-score-left-sum",
      "arena-score-right-max",
      "arena-score-right-sum",
    ]) {
      const el = document.getElementById(id);
      if (el) el.textContent = "—";
    }
  }

  /**
   * @param {{ board: number[][] } | null} leftStep
   * @param {{ board: number[][] } | null} rightStep
   */
  function updateArenaScoreTracker(leftStep, rightStep) {
    function set(prefix, step) {
      const maxEl = document.getElementById("arena-score-" + prefix + "-max");
      const sumEl = document.getElementById("arena-score-" + prefix + "-sum");
      if (!maxEl || !sumEl || !step || !step.board) return;
      maxEl.textContent = String(boardMaxTile(step.board));
      sumEl.textContent = String(boardSum(step.board));
    }
    set("left", leftStep);
    set("right", rightStep);
  }

  /**
   * Order valid moves globally (left before right when both step in one tick) so
   * "first to 2048" matches server / UI step order.
   * @param {{ left?: any, right?: any }} steps
   */
  function arenaApplyStepsWinTracking(steps) {
    if (!arena || arena.scoreFinalized) return;
    const order = [];
    if (steps.left) order.push(["left", steps.left]);
    if (steps.right) order.push(["right", steps.right]);
    for (const [side, st] of order) {
      if (!st || !st.valid) continue;
      arena.globalHalfStep += 1;
      const key = side === "left" ? "leftFirst2048Seq" : "rightFirst2048Seq";
      if (st.won && arena[key] == null) arena[key] = arena.globalHalfStep;
    }
  }

  /** @param {any} pack */
  function syncArenaStepCacheFromPack(pack) {
    if (!arena) return;
    const steps = pack.steps || {};
    if (!arenaStepCache) arenaStepCache = { left: null, right: null };
    arenaApplyStepsWinTracking(steps);
    if (steps.left) arenaStepCache.left = steps.left;
    if (steps.right) arenaStepCache.right = steps.right;
    if (arenaStepCache.left && arenaStepCache.right) {
      updateArenaScoreTracker(arenaStepCache.left, arenaStepCache.right);
      maybeUpdateArenaOutcomeOverlays(arenaStepCache.left, arenaStepCache.right);
    }
  }

  function compareArenaSides(leftBoard, rightBoard) {
    const ml = boardMaxTile(leftBoard);
    const mr = boardMaxTile(rightBoard);
    if (ml !== mr) return ml > mr ? -1 : 1;
    const sl = boardSum(leftBoard);
    const sr = boardSum(rightBoard);
    if (sl !== sr) return sl > sr ? -1 : 1;
    return 0;
  }

  const ARENA_OVERLAY_IDS = { left: "arena-overlay-left", right: "arena-overlay-right" };

  function resetArenaPaneOverlay(side) {
    const id = ARENA_OVERLAY_IDS[side];
    const el = id ? document.getElementById(id) : null;
    if (!el) return;
    el.classList.add("pane-overlay--hidden");
    el.classList.remove(
      "pane-overlay--win",
      "pane-overlay--lose",
      "pane-overlay--tie",
      "pane-overlay--lose-first",
      "pane-overlay--word-only"
    );
    const word = el.querySelector(".pane-overlay__word");
    const sub = el.querySelector(".pane-overlay__sub");
    if (word) word.textContent = "";
    if (sub) sub.textContent = "";
    el.setAttribute("aria-hidden", "true");
  }

  function clearArenaOverlays() {
    resetArenaPaneOverlay("left");
    resetArenaPaneOverlay("right");
  }

  /**
   * @param {"left" | "right"} side
   * @param {"hidden" | "win" | "lose" | "tie" | "lose-first"} kind
   */
  function showArenaPaneOverlay(side, kind) {
    if (kind === "hidden") {
      resetArenaPaneOverlay(side);
      return;
    }
    const el = document.getElementById(ARENA_OVERLAY_IDS[side]);
    if (!el) return;
    resetArenaPaneOverlay(side);
    el.classList.remove("pane-overlay--hidden");
    if (kind === "win") el.classList.add("pane-overlay--win");
    else if (kind === "tie") el.classList.add("pane-overlay--tie");
    else if (kind === "lose-first") el.classList.add("pane-overlay--lose-first");
    else el.classList.add("pane-overlay--lose");
    const wordOnly = kind === "win" || kind === "lose" || kind === "lose-first" || kind === "tie";
    el.classList.toggle("pane-overlay--word-only", wordOnly);
    const word = el.querySelector(".pane-overlay__word");
    const sub = el.querySelector(".pane-overlay__sub");
    if (!word || !sub) return;
    el.setAttribute("aria-hidden", "false");
    if (kind === "win") {
      word.textContent = "WIN";
      sub.textContent = "";
    } else if (kind === "tie") {
      word.textContent = "TIE";
      sub.textContent = "";
    } else if (kind === "lose-first") {
      word.textContent = "LOSE";
      sub.textContent = "";
    } else {
      word.textContent = "LOSE";
      sub.textContent = "";
    }
  }

  /** @param {"left" | "right" | "tie"} winner */
  function showArenaWinnerPair(winner) {
    resetArenaPaneOverlay("left");
    resetArenaPaneOverlay("right");
    if (winner === "left") {
      showArenaPaneOverlay("left", "win");
      showArenaPaneOverlay("right", "lose");
    } else if (winner === "right") {
      showArenaPaneOverlay("right", "win");
      showArenaPaneOverlay("left", "lose");
    } else {
      showArenaPaneOverlay("left", "tie");
      showArenaPaneOverlay("right", "tie");
    }
  }

  function finalizeArenaBy2048Race() {
    if (!arena || arena.scoreFinalized) return;
    const lS = arena.leftFirst2048Seq;
    const rS = arena.rightFirst2048Seq;
    if (lS == null && rS == null) return;
    let winner;
    let msg;
    if (lS != null && rS != null) {
      if (lS < rS) {
        winner = "left";
        msg = "Left wins — reached 2048 first.";
      } else if (rS < lS) {
        winner = "right";
        msg = "Right wins — reached 2048 first.";
      } else {
        winner = "tie";
        msg = "Tie — both reached 2048 on the same global move index.";
      }
    } else if (lS != null) {
      winner = "left";
      msg = "Left wins — reached 2048.";
    } else {
      winner = "right";
      msg = "Right wins — reached 2048.";
    }
    showArenaWinnerPair(winner);
    toastArena.textContent = msg;
    arena.scoreFinalized = true;
  }

  /**
   * Neither lane reached 2048; both boards are dead — compare max tile then sum.
   * @param {{ board: number[][], game_over: boolean, won: boolean }} sl
   * @param {{ board: number[][], game_over: boolean, won: boolean }} sr
   */
  function finalizeArenaByStalemate(sl, sr) {
    if (!arena || arena.scoreFinalized || !sl.board || !sr.board) return;
    const cmp = compareArenaSides(sl.board, sr.board);
    resetArenaPaneOverlay("left");
    resetArenaPaneOverlay("right");
    if (cmp < 0) {
      showArenaPaneOverlay("left", "win");
      showArenaPaneOverlay("right", "lose");
      toastArena.textContent = "Left wins — higher max tile (then higher sum of tiles).";
    } else if (cmp > 0) {
      showArenaPaneOverlay("right", "win");
      showArenaPaneOverlay("left", "lose");
      toastArena.textContent = "Right wins — higher max tile (then higher sum of tiles).";
    } else {
      showArenaPaneOverlay("left", "tie");
      showArenaPaneOverlay("right", "tie");
      toastArena.textContent = "Draw — same max tile and tile sum.";
    }
    arena.scoreFinalized = true;
  }

  /**
   * @param {{ board: number[][], game_over: boolean, won: boolean }} sl
   * @param {{ board: number[][], game_over: boolean, won: boolean }} sr
   */
  function maybeUpdateArenaOutcomeOverlays(sl, sr) {
    if (!arena || !sl || !sr) return;
    if (arena.scoreFinalized) return;
    if (arena.leftFirst2048Seq != null || arena.rightFirst2048Seq != null) {
      finalizeArenaBy2048Race();
      return;
    }
    if (sl.game_over && sr.game_over) {
      finalizeArenaByStalemate(sl, sr);
      return;
    }
    if (sl.game_over && !sr.game_over) {
      if (arena.firstStuck == null) arena.firstStuck = "left";
      showArenaPaneOverlay("left", "lose-first");
    } else if (!sl.game_over && sr.game_over) {
      if (arena.firstStuck == null) arena.firstStuck = "right";
      showArenaPaneOverlay("right", "lose-first");
    }
  }

  function applyModeUI() {
    const m = modeSelect.value;
    sectionSolo.classList.toggle("hidden", m !== "solo");
    sectionVersus.classList.toggle("hidden", m !== "versus");
    sectionArena.classList.toggle("hidden", m !== "arena");
    page.classList.toggle("page--wide", m === "versus" || m === "arena");
    if (m !== "arena") {
      arenaStepCache = null;
      resetArenaScoreTrackerDisplay();
    }
  }

  function arenaAutoResetBtn() {
    const btn = document.getElementById("arena-auto");
    if (btn) btn.textContent = "Auto both bots";
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
  const versusDqnModel = document.getElementById("versus-dqn-model");
  const versusModelWrap = document.getElementById("versus-model-wrap");
  const versusSeed = document.getElementById("versus-seed");

  function syncVersusDqnModelUI() {
    if (!versusModelWrap) return;
    const show = versusOppSel.value === "dqn";
    versusModelWrap.classList.toggle("hidden", !show);
  }
  versusOppSel.addEventListener("change", syncVersusDqnModelUI);
  syncVersusDqnModelUI();

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
    const ck = opponent === "dqn" && versusDqnModel ? versusDqnModel.value : null;
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
      toastVersus.textContent = err.detail || "Could not start duel (DQN needs a .pt under checkpoints/).";
      return;
    }
    const d = await res.json();
    duel = {
      matchId: d.match_id ?? d.matchId,
      playerId: d.left_game_id ?? d.leftGameId,
      botId: d.right_game_id ?? d.rightGameId,
      globalHalfStep: 0,
      youFirst2048Seq: null,
      botFirst2048Seq: null,
      raceToastShown: false,
    };
    if (opponent === "dqn" && versusDqnModel) {
      const label = versusDqnModel.options[versusDqnModel.selectedIndex].text;
      document.getElementById("versus-bot-title").textContent = label + " (DQN)";
    } else {
      document.getElementById("versus-bot-title").textContent = opponent + " bot";
    }

    versusYou.lastBoard = null;
    versusBot.lastBoard = null;
    const [sy, sb] = await Promise.all([fetchSnap(duel.playerId), fetchSnap(duel.botId)]);
    renderInto(versusYou, sy.board, { animate: false });
    renderInto(versusBot, sb.board, { animate: false });
    statusYou.textContent = formatStatus(sy);
    statusBot.textContent = formatStatus(sb);
  }

  function maybeVersusRaceToast() {
    if (!duel || duel.raceToastShown) return;
    const y = duel.youFirst2048Seq;
    const b = duel.botFirst2048Seq;
    if (y == null && b == null) return;
    duel.raceToastShown = true;
    if (y != null && b != null) {
      if (y < b) toastVersus.textContent = "You win the race — you reached 2048 first.";
      else if (b < y) toastVersus.textContent = "Bot wins the race — reached 2048 before you.";
      else toastVersus.textContent = "Tie — both reached 2048 on the same exchange.";
    } else if (y != null) {
      toastVersus.textContent = "You win the race — you reached 2048.";
    } else {
      toastVersus.textContent = "Bot wins the race — reached 2048.";
    }
  }

  async function versusPlayerStep(action) {
    if (!duel) {
      toastVersus.textContent = "Start a duel first.";
      return;
    }
    const res = await fetch(
      "/api/matches/" + encodeURIComponent(duel.matchId) + "/versus-step",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      }
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toastVersus.textContent =
        (typeof err.detail === "string" ? err.detail : null) || "Move failed.";
      return;
    }
    const pack = await res.json();
    const data = pack.player;
    if (data.valid) {
      duel.globalHalfStep += 1;
      if (data.won && duel.youFirst2048Seq == null) duel.youFirst2048Seq = duel.globalHalfStep;
    }
    if (pack.opponent && pack.opponent.valid) {
      duel.globalHalfStep += 1;
      if (pack.opponent.won && duel.botFirst2048Seq == null) duel.botFirst2048Seq = duel.globalHalfStep;
    }
    maybeVersusRaceToast();
    if (!data.valid) {
      shakeWrap(versusYou.wrap);
      statusYou.textContent = formatStatus(data);
      return;
    }
    renderInto(versusYou, data.board, { animate: true });
    statusYou.textContent = formatStatus(data);
    if (pack.opponent) {
      renderInto(versusBot, pack.opponent.board, { animate: true });
      statusBot.textContent = formatStatus(pack.opponent);
    }
  }

  async function opponentTurn() {
    if (!duel) return;
    const res = await fetch(
      "/api/matches/" + encodeURIComponent(duel.matchId) + "/opponent-turn",
      { method: "POST" }
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toastVersus.textContent =
        (typeof err.detail === "string" ? err.detail : null) || "Bot step failed.";
      return;
    }
    const data = await res.json();
    if (data.valid) {
      duel.globalHalfStep += 1;
      if (data.won && duel.botFirst2048Seq == null) duel.botFirst2048Seq = duel.globalHalfStep;
    }
    maybeVersusRaceToast();
    renderInto(versusBot, data.board, { animate: true });
    statusBot.textContent = formatStatus(data);
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
  const arenaLeftModel = document.getElementById("arena-left-model");
  const arenaRightModel = document.getElementById("arena-right-model");

  function syncArenaModelControls() {
    const lw = document.getElementById("arena-left-model-wrap");
    const rw = document.getElementById("arena-right-model-wrap");
    if (lw) lw.classList.toggle("hidden", arenaLeft.value !== "dqn");
    if (rw) rw.classList.toggle("hidden", arenaRight.value !== "dqn");
  }
  arenaLeft.addEventListener("change", syncArenaModelControls);
  arenaRight.addEventListener("change", syncArenaModelControls);

  arenaLeft.value = "dqn";
  arenaRight.value = "dqn";
  if (arenaLeftModel) arenaLeftModel.selectedIndex = 0;
  if (arenaRightModel) arenaRightModel.selectedIndex = 1;
  syncArenaModelControls();

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
      document.getElementById("status-arena-left").textContent = formatArenaStatusOnly(steps.left);
    }
    if (steps.right) {
      renderInto(arenaR, steps.right.board, { animate });
      document.getElementById("status-arena-right").textContent = formatArenaStatusOnly(steps.right);
    }
    syncArenaStepCacheFromPack(pack);
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
    arenaStepCache = null;
    resetArenaScoreTrackerDisplay();
    const left = arenaLeft.value;
    const right = arenaRight.value;
    const body = { mode: "arena", left, right };
    if (left === "dqn" && arenaLeftModel) body.left_checkpoint = arenaLeftModel.value;
    if (right === "dqn" && arenaRightModel) body.right_checkpoint = arenaRightModel.value;
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
    arena = {
      matchId: d.match_id,
      leftId: d.left_game_id,
      rightId: d.right_game_id,
      firstStuck: null,
      scoreFinalized: false,
      globalHalfStep: 0,
      leftFirst2048Seq: null,
      rightFirst2048Seq: null,
    };
    clearArenaOverlays();
    function arenaSideTitle(policySel, modelSel) {
      if (policySel.value === "dqn" && modelSel) {
        return modelSel.options[modelSel.selectedIndex].text + " (DQN)";
      }
      return policySel.options[policySel.selectedIndex].text;
    }
    document.getElementById("arena-left-title").textContent = arenaSideTitle(arenaLeft, arenaLeftModel);
    document.getElementById("arena-right-title").textContent = arenaSideTitle(arenaRight, arenaRightModel);

    arenaL.lastBoard = null;
    arenaR.lastBoard = null;
    const [sl, sr] = await Promise.all([fetchSnap(arena.leftId), fetchSnap(arena.rightId)]);
    renderInto(arenaL, sl.board, { animate: false });
    renderInto(arenaR, sr.board, { animate: false });
    arenaStepCache = { left: sl, right: sr };
    updateArenaScoreTracker(sl, sr);
    document.getElementById("status-arena-left").textContent = formatArenaStatusOnly(sl);
    document.getElementById("status-arena-right").textContent = formatArenaStatusOnly(sr);
    maybeUpdateArenaOutcomeOverlays(sl, sr);
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

        if (arena.scoreFinalized) {
          break;
        }
        if (sl.game_over && sr.game_over) {
          break;
        }

        await sleep(ARENA_AUTO_DELAY_MS);
      }
      if (
        arenaAutoActive &&
        toastArena.textContent === "" &&
        arena &&
        !arena.scoreFinalized
      ) {
        toastArena.textContent =
          "Stopped — safety step limit reached (reload or adjust code if needed).";
      }
    } finally {
      if (arena && !arena.scoreFinalized && toastArena.textContent === "") {
        toastArena.textContent =
          "Stopped early — run auto until the race finishes (2048 or both boards stuck).";
      }
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

  soloBootstrap();
})();
