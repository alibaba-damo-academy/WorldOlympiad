(() => {
  // ===== 1. VERSION TAB SWITCHER (WorldOlympiad 1.0 <-> WorldOlympiad 2.0) =====
  const V1_HASHES = new Set(["#v1", "#overview", "#data", "#leaderboard", "#findings", "#metrics", "#cases", "#acknowledgement", "#license", "#citation"]);

  const versionBtns = Array.from(document.querySelectorAll("[data-version-switch]"));
  const panelV1 = document.getElementById("panel-v1");
  const panelV2 = document.getElementById("panel-v2");
  const heroV1 = document.getElementById("hero-v1");
  const heroV2 = document.getElementById("hero-v2");
  const heroVideoV1 = document.getElementById("heroVideoV1");
  const heroVideoV2 = document.getElementById("heroVideoV2");
  const navV1 = document.getElementById("nav-links-v1");
  const navV2 = document.getElementById("nav-links-v2");
  const tocV1 = document.getElementById("toc-v1");
  const tocV2 = document.getElementById("toc-v2");

  let activeVersion = "v2";

  function pauseVideosIn(el) {
    if (!el) return;
    el.querySelectorAll("video").forEach((v) => {
      try { v.pause(); } catch (_) {}
    });
  }

  function setVersion(ver, updateHash = true) {
    activeVersion = ver === "v1" ? "v1" : "v2";
    const isV2 = activeVersion === "v2";

    panelV1.classList.toggle("hidden", isV2);
    panelV2.classList.toggle("hidden", !isV2);
    heroV1.classList.toggle("hidden", isV2);
    heroV2.classList.toggle("hidden", !isV2);
    navV1.classList.toggle("hidden", isV2);
    navV2.classList.toggle("hidden", !isV2);
    tocV1.classList.toggle("hidden", isV2);
    tocV2.classList.toggle("hidden", !isV2);

    heroVideoV1.classList.toggle("hidden", isV2);
    heroVideoV2.classList.toggle("hidden", !isV2);
    if (isV2) {
      heroVideoV1.pause();
      heroVideoV2.play().catch(() => {});
      pauseVideosIn(panelV1);
      if (typeof window.syncV2VisibleRailVideos === "function") {
        window.syncV2VisibleRailVideos();
      }
    } else {
      heroVideoV2.pause();
      heroVideoV1.play().catch(() => {});
      pauseVideosIn(panelV2);
      // Resume 1.0 scrolling data rail videos
      panelV1.querySelectorAll("#dataShowcase video").forEach((v) => v.play().catch(() => {}));
      if (typeof window.updateActiveCaseFromScroll === "function") {
        window.updateActiveCaseFromScroll();
      }
    }

    versionBtns.forEach((btn) => {
      const match = btn.dataset.versionSwitch === activeVersion;
      btn.classList.toggle("active", match);
      btn.setAttribute("aria-selected", String(match));
    });

    document.title = isV2
      ? "WorldOlympiad 2.0 — Embodied World Model Benchmark"
      : "WorldOlympiad — Video World Model Benchmark";

    if (updateHash) {
      try {
        history.replaceState(null, "", isV2 ? "#v2" : "#v1");
      } catch (_) {}
    }
  }

  versionBtns.forEach((btn) => {
    btn.addEventListener("click", () => setVersion(btn.dataset.versionSwitch, true));
  });

  // ===== 2. LIGHTBOX MODAL (VIDEOS + HIGH-RES FIGURES) =====
  const lightbox = document.getElementById("v2Lightbox");
  const lightboxTitle = document.getElementById("v2LightboxTitle");
  const lightboxBody = document.getElementById("v2LightboxBody");
  const lightboxClose = document.getElementById("v2LightboxClose");

  function openLightbox(type, src, title) {
    lightboxTitle.textContent = title || "";
    if (type === "video") {
      lightboxBody.innerHTML = `<video src="${src}" controls autoplay loop playsinline></video>`;
    } else {
      lightboxBody.innerHTML = `<img src="${src}" alt="${title || ""}">`;
    }
    lightbox.classList.add("open");
    lightbox.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
  }

  function closeLightbox() {
    lightboxBody.innerHTML = "";
    lightbox.classList.remove("open");
    lightbox.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
  }

  lightboxClose?.addEventListener("click", closeLightbox);
  lightbox?.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox.classList.contains("open")) closeLightbox();
  });

  document.querySelectorAll("[data-zoom-img]").forEach((img) => {
    img.addEventListener("click", () => {
      openLightbox("img", img.getAttribute("src"), img.getAttribute("alt") || "Figure");
    });
  });

  // ===== 3. FLOWING MULTI-ROW VIDEO RAILS (39 DEMOS · 1 FLOWING ROW PER CATEGORY) =====
  const V2_RAILS = [
    {
      key: "real",
      title: "Real-Robot Bimanual Suite",
      count: "300 Demos · 10 Tasks",
      hint: "AgileX Piper · Front View",
      dir: "",
      items: [
        { title: "Pack Backpack", src: "assets/v2/video/real/real-pack_backpack.mp4", poster: "assets/v2/video/real/real-pack_backpack.jpg" },
        { title: "Clean Whiteboard", src: "assets/v2/video/real/real-clean_whiteboard.mp4", poster: "assets/v2/video/real/real-clean_whiteboard.jpg" },
        { title: "Pour Balls into Cup", src: "assets/v2/video/real/real-pour_balls.mp4", poster: "assets/v2/video/real/real-pour_balls.jpg" },
        { title: "Build Wooden Tower", src: "assets/v2/video/real/real-build_tower.mp4", poster: "assets/v2/video/real/real-build_tower.jpg" },
        { title: "Arrange Utensils", src: "assets/v2/video/real/real-arrange_utensils.mp4", poster: "assets/v2/video/real/real-arrange_utensils.jpg" },
        { title: "Press Button by Number", src: "assets/v2/video/real/real-press_by_number.mp4", poster: "assets/v2/video/real/real-press_by_number.jpg" },
        { title: "Put Objects into Box", src: "assets/v2/video/real/real-put_objects_in_box.mp4", poster: "assets/v2/video/real/real-put_objects_in_box.jpg" },
        { title: "Put Tools into Cup", src: "assets/v2/video/real/real-put_tools_in_holder.mp4", poster: "assets/v2/video/real/real-put_tools_in_holder.jpg" },
        { title: "Stack Two Baskets", src: "assets/v2/video/real/real-stack_baskets.mp4", poster: "assets/v2/video/real/real-stack_baskets.jpg" },
        { title: "Stack Two Blocks", src: "assets/v2/video/real/real-stack_blocks.mp4", poster: "assets/v2/video/real/real-stack_blocks.jpg" },
        { title: "Pack Backpack (Ep. 2)", src: "assets/v2/video/real/real-pack_backpack-01.mp4", poster: "assets/v2/video/real/real-pack_backpack-01.jpg" },
        { title: "Clean Whiteboard (Ep. 2)", src: "assets/v2/video/real/real-clean_whiteboard-01.mp4", poster: "assets/v2/video/real/real-clean_whiteboard-01.jpg" },
        { title: "Pour Balls (Ep. 2)", src: "assets/v2/video/real/real-pour_balls-01.mp4", poster: "assets/v2/video/real/real-pour_balls-01.jpg" },
        { title: "Build Tower (Ep. 2)", src: "assets/v2/video/real/real-build_tower-01.mp4", poster: "assets/v2/video/real/real-build_tower-01.jpg" },
        { title: "Arrange Utensils (Ep. 2)", src: "assets/v2/video/real/real-arrange_utensils-01.mp4", poster: "assets/v2/video/real/real-arrange_utensils-01.jpg" },
        { title: "Press by Number (Ep. 2)", src: "assets/v2/video/real/real-press_by_number-01.mp4", poster: "assets/v2/video/real/real-press_by_number-01.jpg" },
      ],
    },
    {
      key: "sim",
      title: "Constructed Physics-Checked Tasks",
      count: "200 Tasks · 5 Families",
      hint: "SceneCode · PyBullet · Blender PBR",
      dir: "reverse",
      items: [
        { title: "Open Articulated Cabinet", src: "assets/v2/video/sim/pbr_cabinet.mp4", poster: "assets/v2/video/sim/pbr_cabinet.jpg" },
        { title: "Place Bottle into Basket", src: "assets/v2/video/sim/pbr_basket.mp4", poster: "assets/v2/video/sim/pbr_basket.jpg" },
        { title: "Restore Bottle Upright", src: "assets/v2/video/sim/pbr_lotion.mp4", poster: "assets/v2/video/sim/pbr_lotion.jpg" },
        { title: "Hang Cup on Rack (Panda)", src: "assets/v2/video/sim/cup_hang_hd.mp4", poster: "assets/v2/video/sim/cup_hang_hd.jpg" },
        { title: "Align Spice Jar on Mat", src: "assets/v2/video/sim/pbr_spice.mp4", poster: "assets/v2/video/sim/pbr_spice.jpg" },
        { title: "Cabinet Door (Exterior View)", src: "assets/v2/video/sim/pbr_cabinet_alt.mp4", poster: "assets/v2/video/sim/pbr_cabinet_alt.jpg" },
        { title: "Bottle Basket (Side View)", src: "assets/v2/video/sim/pbr_basket_alt.mp4", poster: "assets/v2/video/sim/pbr_basket_alt.jpg" },
        { title: "Hang Cup (Dual-Arm ARX5)", src: "assets/v2/video/sim/cup_hanging_arx5.mp4", poster: "assets/v2/video/sim/cup_hanging_arx5.jpg" },
        { title: "Cup Uprighting", src: "assets/v2/video/sim/cup_uprighting.mp4", poster: "assets/v2/video/sim/cup_uprighting.jpg" },
        { title: "Tool / Dowel Manipulation", src: "assets/v2/video/sim/tool_use.mp4", poster: "assets/v2/video/sim/tool_use.jpg" },
        { title: "Tissue-Box Pushing", src: "assets/v2/video/sim/tissue_push.mp4", poster: "assets/v2/video/sim/tissue_push.jpg" },
        { title: "Cabinet Articulation (Dual)", src: "assets/v2/video/sim/cabinet.mp4", poster: "assets/v2/video/sim/cabinet.jpg" },
      ],
    },
    {
      key: "pub",
      title: "Public Embodied Benchmarks",
      count: "900 Cases · 5 Sources",
      hint: "RoboDojo · ManiSkill3 · Table30 · RoboCOIN · RoboMIND",
      dir: "medium",
      items: [
        { title: "Insert Three Tubes", src: "assets/v2/video/public/public-three-tubes.mp4", poster: "assets/v2/video/public/public-three-tubes.jpg" },
        { title: "Sort Geometric Toy", src: "assets/v2/video/public/public-toy.mp4", poster: "assets/v2/video/public/public-toy.jpg" },
        { title: "Stack Cubes (ManiSkill3)", src: "assets/v2/video/public/public-stack-cube.mp4", poster: "assets/v2/video/public/public-stack-cube.jpg" },
        { title: "Stack Bowls (RoboDojo)", src: "assets/v2/video/public/public-bowls.mp4", poster: "assets/v2/video/public/public-bowls.jpg" },
        { title: "Arrange Blocks", src: "assets/v2/video/public/public-blocks.mp4", poster: "assets/v2/video/public/public-blocks.jpg" },
        { title: "Peg Insertion", src: "assets/v2/video/public/public-peg.mp4", poster: "assets/v2/video/public/public-peg.jpg" },
        { title: "Network Cable Task", src: "assets/v2/video/public/public-network.mp4", poster: "assets/v2/video/public/public-network.jpg" },
        { title: "Plug Charger", src: "assets/v2/video/public/public-charger.mp4", poster: "assets/v2/video/public/public-charger.jpg" },
        { title: "Pick Cube (ManiSkill3)", src: "assets/v2/video/public/public-pick-cube.mp4", poster: "assets/v2/video/public/public-pick-cube.jpg" },
        { title: "Insert Single Tube", src: "assets/v2/video/public/public-tube.mp4", poster: "assets/v2/video/public/public-tube.jpg" },
        { title: "Stack Bowls (Alt)", src: "assets/v2/video/public/public-bowls-alt.mp4", poster: "assets/v2/video/public/public-bowls-alt.jpg" },
      ],
    },
  ];

  const railsContainer = document.getElementById("v2VideoRails");
  if (railsContainer) {
    railsContainer.innerHTML = V2_RAILS.map((rail) => {
      const doubled = [...rail.items, ...rail.items];
      return `
        <div class="v2-rail-block">
          <div class="v2-rail-head">
            <div class="v2-rail-title">
              <span>${rail.title}</span>
              <span class="v2-rail-count ${rail.key}">${rail.count}</span>
            </div>
            <span class="v2-rail-hint">${rail.hint}</span>
          </div>
          <div class="v2-rail-window">
            <div class="v2-rail-track ${rail.dir}">
              ${doubled.map((item) => `
                <button class="v2-rail-card" type="button" data-src="${item.src}" data-title="${item.title}">
                  <video muted loop playsinline preload="none" poster="${item.poster}" data-vsrc="${item.src}"></video>
                  <span class="v2-rail-cap">${item.title}</span>
                </button>
              `).join("")}
            </div>
          </div>
        </div>
      `;
    }).join("");

    // Smart viewport-only video playback: only decode/play cards actually visible in the viewport
    const railVideos = Array.from(railsContainer.querySelectorAll(".v2-rail-card video"));
    const visibleSet = new Set();
    const ensureSrc = (v) => {
      if (!v.getAttribute("src") && v.dataset.vsrc) {
        v.setAttribute("src", v.dataset.vsrc);
        v.preload = "auto";
      }
    };
    const syncVisible = () => {
      const isV2Active = !panelV2.classList.contains("hidden");
      railVideos.forEach((v) => {
        if (isV2Active && visibleSet.has(v)) {
          ensureSrc(v);
          if (v.paused) v.play().catch(() => {});
        } else if (!v.paused) {
          v.pause();
        }
      });
    };
    window.syncV2VisibleRailVideos = syncVisible;

    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) visibleSet.add(entry.target);
            else visibleSet.delete(entry.target);
          });
          syncVisible();
        },
        { rootMargin: "140px 180px 140px 180px", threshold: 0.01 }
      );
      railVideos.forEach((v) => io.observe(v));
    } else {
      railVideos.forEach((v) => visibleSet.add(v));
      syncVisible();
    }

    railsContainer.addEventListener("click", (e) => {
      const card = e.target.closest(".v2-rail-card");
      if (card) openLightbox("video", card.dataset.src, card.dataset.title);
    });
  }

  // ===== 4. SYNCED CAE DUAL VIDEOS + LIVE SVG TRAJECTORY PLOTS =====
  function wireSyncPair(btnId, vidAId, vidBId, onTimeUpdate) {
    const btn = document.getElementById(btnId);
    const vA = document.getElementById(vidAId);
    const vB = document.getElementById(vidBId);
    if (!btn || !vA || !vB) return;

    btn.addEventListener("click", () => {
      if (vA.paused) {
        vB.currentTime = Math.min(vB.duration || 999, vA.currentTime);
        vA.play().catch(() => {});
        vB.play().catch(() => {});
        btn.textContent = "Pause Sync";
      } else {
        vA.pause();
        vB.pause();
        btn.textContent = "Play Sync";
      }
    });

    if (onTimeUpdate) {
      vA.addEventListener("timeupdate", () => onTimeUpdate(vA.currentTime, vA.duration || 1));
    }
  }

  let caeDrawCursor = null;
  wireSyncPair("caeSyncBanana", "caeBananaRef", "caeBananaCos", (t, dur) => {
    if (caeDrawCursor) caeDrawCursor(t / Math.max(dur, 0.1));
  });
  wireSyncPair("caeSyncBowl", "caeBowlRef", "caeBowlLing");

  fetch("assets/v2/data/cae-banana.json")
    .then((r) => r.json())
    .then((data) => {
      const ref = data.roles[0];
      const cos = data.roles[1];
      const pathSvg = document.getElementById("caePathSvg");
      const velSvg = document.getElementById("caeVelSvg");
      if (!pathSvg || !velSvg) return;

      const allPts = [...ref.path, ...cos.path];
      const ys = allPts.map((p) => p[1]);
      const zs = allPts.map((p) => p[2]);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const minZ = Math.min(...zs), maxZ = Math.max(...zs);
      const mapYZ = (p) => [
        18 + ((p[1] - minY) / Math.max(maxY - minY, 0.05)) * 384,
        124 - ((p[2] - minZ) / Math.max(maxZ - minZ, 0.05)) * 102,
      ];

      const refD = ref.path.map((p, i) => `${i === 0 ? "M" : "L"}${mapYZ(p).map((v) => v.toFixed(1)).join(",")}`).join(" ");
      const cosD = cos.path.map((p, i) => `${i === 0 ? "M" : "L"}${mapYZ(p).map((v) => v.toFixed(1)).join(",")}`).join(" ");

      pathSvg.innerHTML = `
        <line x1="18" y1="124" x2="402" y2="124" stroke="#e2e8f0" stroke-width="1"/>
        <path d="${refD}" fill="none" stroke="#2563eb" stroke-width="2.6" stroke-linecap="round"/>
        <path d="${cosD}" fill="none" stroke="#dc2626" stroke-width="2.4" stroke-dasharray="6 4" stroke-linecap="round"/>
        <circle id="caePathDotRef" cx="${mapYZ(ref.path[0])[0]}" cy="${mapYZ(ref.path[0])[1]}" r="5" fill="#2563eb" stroke="#fff" stroke-width="1.6"/>
        <circle id="caePathDotCos" cx="${mapYZ(cos.path[0])[0]}" cy="${mapYZ(cos.path[0])[1]}" r="5" fill="#dc2626" stroke="#fff" stroke-width="1.6"/>
      `;

      const dur = data.duration || 16.73;
      const maxSp = 0.55;
      const mapSp = (pt) => [
        18 + (pt[0] / dur) * 384,
        124 - Math.min(1, pt[1] / maxSp) * 102,
      ];
      const spRefD = ref.speed.map((pt, i) => `${i === 0 ? "M" : "L"}${mapSp(pt).map((v) => v.toFixed(1)).join(",")}`).join(" ");
      const spCosD = cos.speed.map((pt, i) => `${i === 0 ? "M" : "L"}${mapSp(pt).map((v) => v.toFixed(1)).join(",")}`).join(" ");

      velSvg.innerHTML = `
        <line x1="18" y1="124" x2="402" y2="124" stroke="#e2e8f0" stroke-width="1"/>
        <path d="${spRefD}" fill="none" stroke="#2563eb" stroke-width="2.4" stroke-linecap="round"/>
        <path d="${spCosD}" fill="none" stroke="#dc2626" stroke-width="2.2" stroke-dasharray="6 4" stroke-linecap="round"/>
        <line id="caeVelCursor" x1="18" y1="14" x2="18" y2="126" stroke="#0f172a" stroke-width="1.4" stroke-dasharray="3 3" opacity="0.5"/>
      `;

      caeDrawCursor = (frac) => {
        const clamped = Math.max(0, Math.min(1, frac));
        const idx = Math.min(ref.path.length - 1, Math.floor(clamped * (ref.path.length - 1)));
        const [rx, ry] = mapYZ(ref.path[idx]);
        const [cx, cy] = mapYZ(cos.path[Math.min(cos.path.length - 1, idx)]);
        document.getElementById("caePathDotRef")?.setAttribute("cx", rx.toFixed(1));
        document.getElementById("caePathDotRef")?.setAttribute("cy", ry.toFixed(1));
        document.getElementById("caePathDotCos")?.setAttribute("cx", cx.toFixed(1));
        document.getElementById("caePathDotCos")?.setAttribute("cy", cy.toFixed(1));
        const vx = (18 + clamped * 384).toFixed(1);
        document.getElementById("caeVelCursor")?.setAttribute("x1", vx);
        document.getElementById("caeVelCursor")?.setAttribute("x2", vx);
      };
    })
    .catch(() => {});

  // ===== 5. UNIFIED 5-MODEL LEADERBOARD + 11-DIM RADAR + FINE-GRAINED BREAKDOWNS =====
  const V2_MODELS = [
    {
      name: "GR00T N1.6", short: "GR00T N1.6", color: "#dc2626",
      all: 0.643, p: 0.920, g: 0.547, i: 0.460, path: 0.239, vel: 0.146, human: 0.595,
      dims: [0.915, 0.932, 0.537, 0.510, 0.903, 0.239, 0.531, 0.208, 0.736, 0.523, 0.146],
      families: [0.585, 0.620, 0.513, 0.592, 0.558],
      pubAll: 0.660, conAll: 0.574, pubPath: 0.237, conPath: 0.246,
    },
    {
      name: "LingBot-VA", short: "LingBot-VA", color: "#0f9f6e",
      all: 0.519, p: 0.735, g: 0.388, i: 0.433, path: 0.246, vel: 0.159, human: 0.608,
      dims: [0.744, 0.732, 0.374, 0.377, 0.556, 0.246, 0.463, 0.549, 0.546, 0.388, 0.159],
      families: [0.628, 0.597, 0.537, 0.529, 0.621],
      pubAll: 0.504, conAll: 0.580, pubPath: 0.261, conPath: 0.184,
    },
    {
      name: "Long-WAM", short: "Long-WAM", color: "#d97706",
      all: 0.496, p: 0.726, g: 0.390, i: 0.373, path: 0.243, vel: 0.157, human: 0.512,
      dims: [0.748, 0.691, 0.365, 0.363, 0.588, 0.243, 0.403, 0.369, 0.510, 0.361, 0.157],
      families: [0.616, 0.523, 0.411, 0.544, 0.635],
      pubAll: 0.485, conAll: 0.542, pubPath: 0.250, conPath: 0.214,
    },
    {
      name: "Fast-WAM", short: "Fast-WAM", color: "#2563eb",
      all: 0.454, p: 0.702, g: 0.340, i: 0.320, path: 0.242, vel: 0.156, human: 0.487,
      dims: [0.708, 0.662, 0.315, 0.320, 0.484, 0.242, 0.365, 0.244, 0.431, 0.359, 0.156],
      families: [0.598, 0.533, 0.372, 0.537, 0.604],
      pubAll: 0.436, conAll: 0.526, pubPath: 0.260, conPath: 0.171,
    },
    {
      name: "Cosmos3", short: "Cosmos3", color: "#7c3aed",
      all: 0.439, p: 0.685, g: 0.342, i: 0.289, path: 0.382, vel: 0.183, human: 0.435,
      dims: [0.774, 0.577, 0.372, 0.350, 0.265, 0.382, 0.394, 0.235, 0.231, 0.421, 0.183],
      families: [0.328, 0.254, 0.462, 0.308, 0.348],
      pubAll: 0.463, conAll: 0.340, pubPath: 0.434, conPath: 0.174,
    },
  ];

  const V2_AXES = ["Mech", "Mat", "Recon", "Meta", "Cam", "Path", "Local", "Trans", "Global", "Sem", "Vel"];
  const V2_FAMILIES = ["Cabinet", "Upright", "Hanging", "Tool", "Push"];

  // Sortable V2 Leaderboard
  let v2SortKey = "all";
  let v2SortDesc = true;
  const v2Table = document.getElementById("v2LeaderboardTable");
  const v2Tbody = v2Table?.querySelector("tbody");

  function barTd(val, colorVar, strong = false) {
    return `<td class="score-cell${strong ? " score-strong" : ""}"><div class="score-viz" style="--score:${val};--bar-color:${colorVar}"><b>${val.toFixed(3)}</b><span class="score-bar"><i></i></span></div></td>`;
  }

  function renderV2Leaderboard() {
    if (!v2Tbody) return;
    const sorted = [...V2_MODELS].sort((a, b) =>
      v2SortDesc ? b[v2SortKey] - a[v2SortKey] : a[v2SortKey] - b[v2SortKey]
    );
    v2Tbody.innerHTML = sorted.map((m, idx) => {
      const rank = idx + 1;
      const tone = rank === 1 ? "gold" : rank === 2 ? "silver" : rank === 3 ? "bronze" : "";
      return `
        <tr data-rank="${rank}">
          <td class="rank ${tone}"><span class="rank-badge">${rank}</span></td>
          <td><div class="model-cell"><span class="v2-dot" style="background:${m.color}"></span><span class="model-name">${m.name}</span></div></td>
          ${barTd(m.all, "var(--blue)", true)}
          ${barTd(m.p, "var(--orange)")}
          ${barTd(m.g, "var(--violet)")}
          ${barTd(m.i, "var(--green)")}
          ${barTd(m.path, "#7c3aed")}
          ${barTd(m.vel, "#0891b2")}
          ${barTd(m.human, "#0f9f6e")}
        </tr>
      `;
    }).join("");
  }

  v2Table?.querySelectorAll("th[data-v2-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.v2Sort;
      if (v2SortKey === key) v2SortDesc = !v2SortDesc;
      else { v2SortKey = key; v2SortDesc = true; }
      renderV2Leaderboard();
    });
  });
  renderV2Leaderboard();

  // Interactive 11-Dimension Radar
  const radarSvg = document.getElementById("v2RadarSvg");
  const radarPills = document.getElementById("v2RadarPills");
  let spotlightModel = null;

  function renderV2Radar() {
    if (!radarSvg) return;
    const cx = 175, cy = 162, maxR = 116;
    const pt = (i, r) => {
      const a = -Math.PI / 2 + (i / V2_AXES.length) * Math.PI * 2;
      return [cx + Math.cos(a) * r * maxR, cy + Math.sin(a) * r * maxR];
    };

    let html = "";
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach((r) => {
      const pts = V2_AXES.map((_, i) => pt(i, r).map((v) => v.toFixed(1)).join(",")).join(" ");
      html += `<polygon points="${pts}" fill="none" stroke="#e5e9f0" stroke-width="1"/>`;
    });

    V2_AXES.forEach((label, i) => {
      const [ex, ey] = pt(i, 1);
      const [lx, ly] = pt(i, 1.2);
      const isCAE = label === "Path" || label === "Vel";
      html += `<line x1="${cx}" y1="${cy}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" stroke="#e5e9f0" stroke-width="1"/>`;
      html += `<text x="${lx.toFixed(1)}" y="${(ly + 4).toFixed(1)}" text-anchor="middle" fill="${isCAE ? "#7c3aed" : "#64748b"}" font-size="${isCAE ? "11" : "10"}" font-weight="${isCAE ? "800" : "600"}">${label}</text>`;
    });

    V2_MODELS.forEach((m) => {
      const active = !spotlightModel || spotlightModel === m.name;
      const pts = m.dims.map((v, i) => pt(i, v).map((n) => n.toFixed(1)).join(",")).join(" ");
      html += `<polygon points="${pts}" fill="${m.color}" fill-opacity="${active ? (spotlightModel ? 0.22 : 0.08) : 0.015}" stroke="${m.color}" stroke-width="${spotlightModel === m.name ? 2.8 : active ? 2 : 1}" stroke-opacity="${active ? 1 : 0.22}" stroke-linejoin="round"/>`;
    });

    radarSvg.innerHTML = html;
  }

  if (radarPills) {
    radarPills.innerHTML = `<button class="v2-model-pill active" data-m="">All</button>` +
      V2_MODELS.map((m) => `<button class="v2-model-pill" data-m="${m.name}"><i style="background:${m.color}"></i>${m.short}</button>`).join("");
    radarPills.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        spotlightModel = btn.dataset.m || null;
        radarPills.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
        renderV2Radar();
      });
    });
  }
  renderV2Radar();

  // Fine-Grained Breakdown Tabs (11 Sub-Metrics | 5 Task Families | Public vs Constructed)
  const breakdownBox = document.getElementById("v2BreakdownStage");
  const breakdownBtns = Array.from(document.querySelectorAll("#v2BreakdownTabs button"));

  function renderBreakdown(mode) {
    if (!breakdownBox) return;
    if (mode === "submetrics") {
      const colMax = V2_AXES.map((_, ci) => Math.max(...V2_MODELS.map((m) => m.dims[ci])));
      const colTone = ["#d97706", "#d97706", "#7c3aed", "#7c3aed", "#7c3aed", "#6d28d9", "#0f9f6e", "#0f9f6e", "#0f9f6e", "#0891b2", "#6d28d9"];
      breakdownBox.innerHTML = `
        <div class="v2-matrix-wrap">
          <table class="v2-heat-table">
            <thead>
              <tr>
                <th></th>
                <th class="grp" colspan="2" style="color:#d97706;border-color:#fdba74">Physical</th>
                <th class="grp" colspan="4" style="color:#7c3aed;border-color:#c4b5fd">Geometry</th>
                <th class="grp" colspan="5" style="color:#0f9f6e;border-color:#6ee7b7">Interaction</th>
              </tr>
              <tr>
                <th style="text-align:left">Model</th>
                ${V2_AXES.map((a, i) => `<th style="color:${colTone[i]}">${a}</th>`).join("")}
              </tr>
            </thead>
            <tbody>
              ${V2_MODELS.map((m) => `
                <tr>
                  <td><div class="v2-heat-model"><span class="v2-dot" style="background:${m.color}"></span>${m.short}</div></td>
                  ${m.dims.map((val, ci) => {
                    const best = val >= colMax[ci] - 0.0005;
                    const bgAlpha = (0.06 + val * 0.26).toFixed(2);
                    const bg = best ? `${colTone[ci]}24` : `rgba(37,99,235,${(val * 0.16).toFixed(2)})`;
                    return `<td><div class="v2-heat-cell${best ? " best" : ""}" style="background:${bg};color:${best ? colTone[ci] : "#1e293b"}">${val.toFixed(2)}</div></td>`;
                  }).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    } else if (mode === "families") {
      const famMax = V2_FAMILIES.map((_, fi) => Math.max(...V2_MODELS.map((m) => m.families[fi])));
      breakdownBox.innerHTML = `
        <div class="v2-matrix-wrap">
          <table class="v2-heat-table" style="min-width:560px">
            <thead>
              <tr>
                <th style="text-align:left">Model</th>
                ${V2_FAMILIES.map((f) => `<th>${f}</th>`).join("")}
              </tr>
            </thead>
            <tbody>
              ${V2_MODELS.map((m) => `
                <tr>
                  <td><div class="v2-heat-model"><span class="v2-dot" style="background:${m.color}"></span>${m.short}</div></td>
                  ${m.families.map((val, fi) => {
                    const best = val >= famMax[fi] - 0.0005;
                    const bg = best ? `${m.color}20` : `rgba(15,23,42,${(0.02 + val * 0.08).toFixed(2)})`;
                    return `<td><div class="v2-heat-cell${best ? " best" : ""}" style="background:${bg};color:${best ? m.color : "#1e293b"}">${val.toFixed(3)}</div></td>`;
                  }).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    } else {
      breakdownBox.innerHTML = `
        <div class="v2-split-grid">
          <div class="v2-split-card">
            <h4><span>Overall Score (All)</span><span style="font-size:11px;font-weight:700"><span style="color:#2563eb">■ Public (400)</span> &nbsp;<span style="color:#0f9f6e">■ Constructed (100)</span></span></h4>
            ${V2_MODELS.map((m) => `
              <div class="v2-split-row">
                <b>${m.short}</b>
                <div class="v2-split-bars">
                  <div class="v2-split-track"><div class="v2-split-fill" style="width:${(m.pubAll / 0.72 * 100).toFixed(1)}%;background:#2563eb"></div></div>
                  <div class="v2-split-track"><div class="v2-split-fill" style="width:${(m.conAll / 0.72 * 100).toFixed(1)}%;background:#0f9f6e"></div></div>
                </div>
                <span style="text-align:right"><b style="color:#2563eb">${m.pubAll.toFixed(3)}</b> / <b style="color:#0f9f6e">${m.conAll.toFixed(3)}</b></span>
              </div>
            `).join("")}
          </div>
          <div class="v2-split-card">
            <h4><span>CAE Path Fidelity (S_path)</span><span style="font-size:11px;font-weight:700"><span style="color:#7c3aed">■ Public</span> &nbsp;<span style="color:#d97706">■ Constructed</span></span></h4>
            ${V2_MODELS.map((m) => `
              <div class="v2-split-row">
                <b>${m.short}</b>
                <div class="v2-split-bars">
                  <div class="v2-split-track"><div class="v2-split-fill" style="width:${(m.pubPath / 0.5 * 100).toFixed(1)}%;background:#7c3aed"></div></div>
                  <div class="v2-split-track"><div class="v2-split-fill" style="width:${(m.conPath / 0.5 * 100).toFixed(1)}%;background:#d97706"></div></div>
                </div>
                <span style="text-align:right"><b style="color:#7c3aed">${m.pubPath.toFixed(3)}</b> / <b style="color:#d97706">${m.conPath.toFixed(3)}</b></span>
              </div>
            `).join("")}
          </div>
        </div>
      `;
    }
  }

  breakdownBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      breakdownBtns.forEach((b) => b.classList.toggle("active", b === btn));
      renderBreakdown(btn.dataset.v2Breakdown);
    });
  });
  renderBreakdown("submetrics");

  // Qualitative Rollout Figure Tabs
  const rolloutImg = document.getElementById("v2RolloutFig");
  const rolloutBtns = Array.from(document.querySelectorAll("#v2RolloutTabs button"));
  const ROLLOUT_FIGS = {
    dualarm: { src: "assets/v2/fig/rollouts1.png", alt: "Dual-Arm Block Stacking — Fast-WAM vs Long-WAM vs LingBot-VA" },
    constructed: { src: "assets/v2/fig/constructed_gallery_model_comparisons.png", alt: "Constructed Tasks — Cosmos3 vs GR00T" },
    cae: { src: "assets/v2/fig/fig4_particles_path_speed.png", alt: "Continuous Action Trajectories — Reference 30fps vs LingBot-VA 10fps" },
  };
  rolloutBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const item = ROLLOUT_FIGS[btn.dataset.v2Rollout];
      if (!item || !rolloutImg) return;
      rolloutBtns.forEach((b) => b.classList.toggle("active", b === btn));
      rolloutImg.src = item.src;
      rolloutImg.alt = item.alt;
    });
  });

  // Copy V2 BibTeX
  const copyV2Btn = document.getElementById("copyBibtexV2");
  if (copyV2Btn) {
    copyV2Btn.addEventListener("click", async () => {
      const txt = document.getElementById("bibtexBlockV2")?.textContent.trim() || "";
      try {
        await navigator.clipboard.writeText(txt);
        copyV2Btn.textContent = "Copied";
        setTimeout(() => { copyV2Btn.textContent = "Copy"; }, 1600);
      } catch (_) {}
    });
  }

  // V2 TOC Active Highlight
  const v2Sections = Array.from(document.querySelectorAll("#panel-v2 section[id]"));
  const v2TocLinks = Array.from(document.querySelectorAll("#toc-v2 a"));
  const v2Obs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      v2TocLinks.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === "#" + entry.target.id);
      });
    });
  }, { rootMargin: "-30% 0px -55% 0px", threshold: 0 });
  v2Sections.forEach((s) => v2Obs.observe(s));

  // Initial Version Selection (defaults to WorldOlympiad 2.0 unless URL explicitly requests #v1 or a 1.0 anchor)
  const initialHash = window.location.hash;
  setVersion(V1_HASHES.has(initialHash) ? "v1" : "v2", false);
})();
