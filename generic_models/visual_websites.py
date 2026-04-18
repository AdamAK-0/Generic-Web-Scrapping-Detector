"""Visual HTML renderers for the four generic test websites."""

from __future__ import annotations

import hashlib
import html

from generic_models.site_catalog import PageSpec, WebsiteSpec, root_path


def render_visual_page(spec: WebsiteSpec, page: PageSpec) -> str:
    """Render a realistic-looking page while preserving the graph links."""

    nav = _nav(spec)
    content = _site_content(spec, page)
    transitions = _transition_panel(page)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_e(page.title)} - {_e(spec.name)}</title>
  <style>{_css(spec)}</style>
</head>
<body class="{_e(spec.site_id)}">
  <div class="ambient-layer" aria-hidden="true"><i></i><i></i><i></i></div>
  <header class="site-nav">
    <a class="brand" href="{_e(root_path(spec.site_id))}">
      <span class="brand-mark"></span>
      <span>{_e(spec.name)}</span>
    </a>
    <nav>{nav}</nav>
    <span class="graph-pill">{_e(spec.shape)}</span>
  </header>
  <main>
    {content}
    {_experience_strip(spec, page)}
    {transitions}
  </main>
  <script>{_telemetry_script()}</script>
</body>
</html>"""


def _site_content(spec: WebsiteSpec, page: PageSpec) -> str:
    if spec.site_id == "atlas_shop":
        return _atlas_shop_content(spec, page)
    if spec.site_id == "deep_docs":
        return _deep_docs_content(spec, page)
    if spec.site_id == "news_mesh":
        return _news_mesh_content(spec, page)
    if spec.site_id == "support_funnel":
        return _support_content(spec, page)
    return _generic_content(spec, page)


def _atlas_shop_content(spec: WebsiteSpec, page: PageSpec) -> str:
    products = [link for link in page.links if link.startswith("/p/")]
    if page.category == "detail":
        price = 49 + _stable_number(page.path, 420)
        specs = "".join(f"<li>{label}: <strong>{value}</strong></li>" for label, value in _product_specs(page.title))
        return f"""
        <section class="commerce-hero">
          <div>
            <p class="kicker">Field-tested catalog</p>
            <h1>{_e(page.title)}</h1>
            <p>{_e(page.body)} Designed for teams that need reliable water, sensor, and field hardware.</p>
            <div class="price-row"><strong>${price}</strong><a class="button" href="/cart">Add to cart</a><a class="button ghost" href="/catalog">Back to catalog</a></div>
          </div>
          <div class="product-stage">
            <div class="product-orb">{_e(page.title[:2].upper())}</div>
            <div class="rating">4.{_stable_number(page.path, 8) + 1} stars / In stock / Ships today</div>
          </div>
        </section>
        <section class="two-col">
          <article class="panel"><h2>Product Details</h2><p>{_e(page.body)} The page includes real storefront cues: pricing, stock state, product copy, and cart paths.</p></article>
          <article class="panel"><h2>Specifications</h2><ul class="spec-list">{specs}</ul></article>
        </section>"""
    if page.category == "cart":
        return f"""
        <section class="commerce-hero compact">
          <div><p class="kicker">Checkout preview</p><h1>Your field kit cart</h1><p>Review saved items, compare bundles, and continue browsing through realistic commerce paths.</p></div>
          <div class="checkout-card"><strong>Estimated total</strong><span>$487</span><a class="button" href="/catalog">Continue shopping</a></div>
        </section>
        <section class="cards">{_commerce_cards(["Aqua Core", "Flow Guard", "Starter Pack"], cart=True)}</section>"""
    if page.category in {"listing", "home", "utility"}:
        visible_products = products or ["/p/aqua", "/p/flow", "/p/starter", "/p/filter", "/p/pack", "/p/meter"]
        return f"""
        <section class="commerce-hero">
          <div>
            <p class="kicker">Atlas procurement desk</p>
            <h1>{_e(page.title)}</h1>
            <p>{_e(page.body)} Browse categories, compare hardware, search the catalog, and move naturally between product families.</p>
            <div class="search-box"><span>Search field gear, sensors, water kits...</span><a href="/search">Search</a></div>
          </div>
          <div class="promo-card"><span>Spring field bundle</span><strong>Save 18%</strong><a href="/deals">View deals</a></div>
        </section>
        <section class="cards">{_commerce_cards([_title_for_path(path) for path in visible_products[:6]])}</section>"""
    return _info_page(page, "Atlas Shop")


def _deep_docs_content(spec: WebsiteSpec, page: PageSpec) -> str:
    sidebar = "".join(f'<a href="{_e(link)}">{_e(_title_for_path(link))}</a>' for link in _docs_sidebar_links(spec, page))
    if page.category in {"home", "listing"}:
        cards = "".join(
            f'<a class="doc-card" href="{_e(link)}"><span>{_e(_title_for_path(link))}</span><small>{_docs_teaser(link)}</small></a>'
            for link in page.links[:6]
        )
        return f"""
        <section class="docs-hero">
          <div><p class="kicker">Developer documentation</p><h1>{_e(page.title)}</h1><p>{_e(page.body)} Deep Docs behaves like a real documentation portal with nested paths, references, and long reading chains.</p></div>
          <div class="terminal"><span>$ pip install detector-kit</span><span>$ wsd graph audit --prefix 5</span><span>ready: early navigation scoring</span></div>
        </section>
        <section class="doc-grid">{cards}</section>"""
    return f"""
    <section class="docs-layout">
      <aside class="docs-sidebar"><strong>On this portal</strong>{sidebar}</aside>
      <article class="docs-article">
        <p class="kicker">Reference page</p>
        <h1>{_e(page.title)}</h1>
        <p>{_e(page.body)} This article includes headings, code blocks, callouts, and cross-links so browsing looks like a documentation session rather than a flat demo page.</p>
        <h2>Concept</h2>
        <p>The detector treats every page as a node and every click as a directed edge. Prefixes are scored as soon as enough events are available.</p>
        <pre><code>prefix = session.paths[:5]
score = model.predict(graph_features(prefix))</code></pre>
        <h2>Operational notes</h2>
        <ul class="spec-list"><li>Prefer stable paths for graph construction.</li><li>Track branch choices and revisit behavior.</li><li>Compare early prefixes against full-session baselines.</li></ul>
      </article>
    </section>"""


def _news_mesh_content(spec: WebsiteSpec, page: PageSpec) -> str:
    if page.category in {"home", "listing"}:
        story_links = [link for link in page.links if link.startswith("/story/")] or ["/story/water-1", "/story/security-2", "/story/research-3"]
        feature_cards = "".join(
            f'<a class="story-card" href="{_e(link)}"><span>{_e(_topic_for_story(link))}</span><strong>{_e(_headline_for(link))}</strong><small>Analysis / 5 min read</small></a>'
            for link in story_links[:6]
        )
        topic_links = "".join(f'<a class="topic-chip" href="{_e(link)}">{_e(_title_for_path(link))}</a>' for link in page.links if "/topic/" in link)
        return f"""
        <section class="news-hero">
          <div class="masthead">
            <p class="kicker">Independent field intelligence</p>
            <h1>{_e(page.title)}</h1>
            <p>{_e(page.body)} A dense editorial mesh encourages related-story jumps, topic exploration, and non-linear human reading paths.</p>
          </div>
          <aside class="briefing"><strong>Morning briefing</strong><span>Water systems, policy risk, field security, research, and markets.</span>{topic_links}</aside>
        </section>
        <section class="story-grid">{feature_cards}</section>"""
    related = "".join(f'<a href="{_e(link)}">{_e(_headline_for(link))}</a>' for link in page.links if link.startswith("/story/"))
    return f"""
    <section class="article-shell">
      <article class="news-article">
        <p class="kicker">{_e(_topic_for_story(page.path))} desk</p>
        <h1>{_e(_headline_for(page.path))}</h1>
        <p class="byline">By News Mesh Lab / Updated this morning</p>
        <p>{_e(page.body)} The story page includes realistic article length and internal recommendations, making the graph feel like a real publication.</p>
        <p>Human readers may open a related topic, return to trending, pause on an article, or jump laterally across the mesh. A crawler may systematically harvest every story link.</p>
        <blockquote>Navigation structure matters because the same page count can hide very different traversal intent.</blockquote>
        <p>That difference is exactly what graph and entropy features try to capture during early prefixes.</p>
      </article>
      <aside class="related"><strong>Related reading</strong>{related or '<span>No related stories listed.</span>'}</aside>
    </section>"""


def _support_content(spec: WebsiteSpec, page: PageSpec) -> str:
    if page.category in {"home", "listing", "utility"}:
        cards = "".join(
            f'<a class="support-card" href="{_e(link)}"><span>{_support_icon(link)}</span><strong>{_e(_title_for_path(link))}</strong><small>{_support_teaser(link)}</small></a>'
            for link in page.links[:6]
        )
        return f"""
        <section class="support-hero">
          <div>
            <p class="kicker">Customer support center</p>
            <h1>{_e(page.title)}</h1>
            <p>{_e(page.body)} This site is intentionally funnel-shaped: users search, narrow their issue, follow steps, and often end at contact or status pages.</p>
            <div class="support-search">Describe your issue, device, invoice, or login problem</div>
          </div>
          <div class="status-stack"><span>Systems: operational</span><span>Average wait: 3 min</span><span>Help articles: 48</span></div>
        </section>
        <section class="support-grid">{cards}</section>"""
    steps = "".join(f"<li>{text}</li>" for text in ["Check the account state", "Follow the guided step", "Return to help if the path is wrong", "Contact support if unresolved"])
    return f"""
    <section class="support-article">
      <article class="panel">
        <p class="kicker">Guided help article</p>
        <h1>{_e(page.title)}</h1>
        <p>{_e(page.body)} Support navigation creates backtracking and funnel behavior, which tests whether generic features work outside catalog-style sites.</p>
        <ol class="steps">{steps}</ol>
      </article>
      <aside class="contact-panel"><strong>Need help?</strong><input placeholder="Email address" /><textarea placeholder="Short description"></textarea><a class="button" href="/contact">Open contact page</a></aside>
    </section>"""


def _generic_content(spec: WebsiteSpec, page: PageSpec) -> str:
    return f"<section class=\"panel\"><h1>{_e(page.title)}</h1><p>{_e(page.body)}</p></section>"


def _transition_panel(page: PageSpec) -> str:
    links = "".join(f'<a class="link-card" href="{_e(link)}">{_e(_title_for_path(link))}<small>{_e(link)}</small></a>' for link in page.links)
    return f"""
    <section class="transition-panel">
      <div><p class="kicker">Graph transitions</p><h2>Where this page links next</h2><p>These links are the directed edges used by the generic graph model.</p></div>
      <div class="link-grid">{links}</div>
    </section>"""


def _experience_strip(spec: WebsiteSpec, page: PageSpec) -> str:
    """Small visual polish layer that also explains what the detector observes."""

    link_count = len(page.links)
    category = page.category.replace("_", " ").title()
    return f"""
    <section class="experience-strip">
      <article>
        <span>Page role</span>
        <strong>{_e(category)}</strong>
        <small>Structural node type used by the generic graph model.</small>
      </article>
      <article>
        <span>Outgoing choices</span>
        <strong>{link_count}</strong>
        <small>Branching pressure for entropy and transition analysis.</small>
      </article>
      <article>
        <span>Graph shape</span>
        <strong>{_e(spec.shape)}</strong>
        <small>Same edges as the research graph, presented as a real site.</small>
      </article>
    </section>"""


def _nav(spec: WebsiteSpec) -> str:
    root = root_path(spec.site_id)
    root_page = next(page for page in spec.pages if page.path == root)
    links = [root, *list(root_page.links[:5])]
    return "".join(f'<a href="{_e(link)}">{_e("Home" if link == root else _title_for_path(link))}</a>' for link in links)


def _commerce_cards(names: list[str], *, cart: bool = False) -> str:
    cards = []
    for name in names:
        path = "/cart" if cart else f"/p/{_slug(name)}"
        price = 49 + _stable_number(name, 420)
        cards.append(
            f"""<a class="product-card" href="{_e(path)}">
              <span class="product-image">{_e(name[:2].upper())}</span>
              <strong>{_e(name)}</strong>
              <small>Rugged kit / lab verified</small>
              <b>${price}</b>
            </a>"""
        )
    return "".join(cards)


def _info_page(page: PageSpec, brand: str) -> str:
    return f"""
    <section class="panel">
      <p class="kicker">{_e(brand)}</p>
      <h1>{_e(page.title)}</h1>
      <p>{_e(page.body)} This page gives human visitors informational alternatives to direct product browsing.</p>
    </section>"""


def _docs_sidebar_links(spec: WebsiteSpec, page: PageSpec) -> list[str]:
    root = root_path(spec.site_id)
    preferred = ["/docs", "/docs/start", "/docs/install", "/docs/advanced", "/api", "/api/reference", "/tutorials", "/search"]
    seen = []
    for link in [root, *preferred, *page.links]:
        if link not in seen:
            seen.append(link)
    return seen[:10]


def _product_specs(title: str) -> list[tuple[str, str]]:
    seed = _stable_number(title, 100)
    return [
        ("Battery life", f"{8 + seed % 20} hours"),
        ("Warranty", f"{1 + seed % 4} years"),
        ("Field rating", ["IP54", "IP65", "IP67"][seed % 3]),
        ("Weight", f"{1 + seed % 8}.{seed % 9} kg"),
    ]


def _title_for_path(path: str) -> str:
    clean = path.strip("/")
    if not clean:
        return "Home"
    part = clean.split("/")[-1]
    return part.replace("-", " ").replace("_", " ").title()


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")


def _stable_number(text: str, modulo: int) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _docs_teaser(path: str) -> str:
    if "api" in path:
        return "Reference material, events, and model contracts."
    if "advanced" in path:
        return "Tuning, audit, and deployment guidance."
    if "install" in path:
        return "Setup instructions for multiple environments."
    return "Concepts, examples, and tutorial paths."


def _topic_for_story(path: str) -> str:
    slug = path.strip("/").split("/")[-1]
    return slug.split("-", 1)[0].title()


def _headline_for(path: str) -> str:
    topic = _topic_for_story(path)
    return f"{topic} teams adapt as navigation signals reshape the field"


def _support_icon(path: str) -> str:
    if "billing" in path:
        return "$"
    if "account" in path or "login" in path:
        return "@"
    if "device" in path:
        return "#"
    if "status" in path:
        return "!"
    return "?"


def _support_teaser(path: str) -> str:
    if "contact" in path:
        return "Escalate to a human support path."
    if "status" in path:
        return "Check outages and service availability."
    if "search" in path:
        return "Find articles across the support graph."
    return "Follow guided troubleshooting steps."


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _telemetry_script() -> str:
    """Client-side interaction telemetry used only by the local generic demo."""

    return r"""
(() => {
  const sidKey = "generic_wsd_sid";
  const makeId = () => (window.crypto && crypto.randomUUID ? crypto.randomUUID() : "id-" + Date.now() + "-" + Math.random());
  const pageId = makeId();
  const sid = sessionStorage.getItem(sidKey) || makeId();
  sessionStorage.setItem(sidKey, sid);
  let buffer = [];
  let lastMove = 0;
  let lastScroll = 0;

  function normalizeHref(href) {
    if (!href) return null;
    try { return new URL(href, location.href).pathname; }
    catch (_) { return href; }
  }

  function log(type, extra = {}) {
    buffer.push({
      sid,
      page_id: pageId,
      type,
      ts: Date.now(),
      path: location.pathname,
      title: document.title,
      ...extra
    });
    if (buffer.length >= 35) flush();
  }

  function flush() {
    if (!buffer.length) return;
    const payload = JSON.stringify(buffer.splice(0, buffer.length));
    if (navigator.sendBeacon) {
      navigator.sendBeacon("/telemetry", payload);
      return;
    }
    fetch("/telemetry", { method: "POST", headers: {"Content-Type": "application/json"}, body: payload, keepalive: true }).catch(() => {});
  }

  log("page_load", {
    referrer: document.referrer || "",
    interactive_count: document.querySelectorAll("a,button,input,textarea,select").length,
    content_length: document.body.innerText.length
  });

  document.addEventListener("mousemove", event => {
    const now = Date.now();
    if (now - lastMove < 220) return;
    lastMove = now;
    const target = event.target.closest ? event.target.closest("a,button,input,textarea,select") : null;
    log("mousemove", { x: event.clientX, y: event.clientY, target_tag: target ? target.tagName : null, target_href: target && target.href ? normalizeHref(target.href) : null });
  }, { passive: true });

  document.addEventListener("pointerdown", event => {
    const target = event.target.closest ? event.target.closest("a,button,input,textarea,select") : null;
    log("pointerdown", { x: event.clientX, y: event.clientY, target_tag: target ? target.tagName : event.target.tagName, target_href: target && target.href ? normalizeHref(target.href) : null });
  }, true);

  document.addEventListener("pointerup", event => {
    const target = event.target.closest ? event.target.closest("a,button,input,textarea,select") : null;
    log("pointerup", { x: event.clientX, y: event.clientY, target_tag: target ? target.tagName : event.target.tagName, target_href: target && target.href ? normalizeHref(target.href) : null });
  }, true);

  document.addEventListener("click", event => {
    const link = event.target.closest ? event.target.closest("a") : null;
    log("click", {
      x: event.clientX,
      y: event.clientY,
      tag: event.target.tagName,
      href: link ? normalizeHref(link.href || link.getAttribute("href")) : null,
      text: (event.target.innerText || event.target.value || "").slice(0, 90)
    });
    flush();
  }, true);

  document.addEventListener("scroll", () => {
    const now = Date.now();
    if (now - lastScroll < 260) return;
    lastScroll = now;
    log("scroll", { y: window.scrollY, ratio: Math.round((window.scrollY / Math.max(1, document.body.scrollHeight - innerHeight)) * 1000) / 1000 });
  }, { passive: true });

  document.addEventListener("keydown", event => log("keydown", { key: event.key, tag: event.target.tagName }), true);
  document.addEventListener("focusin", event => log("focus", { tag: event.target.tagName, id: event.target.id || null, name: event.target.name || null }), true);
  window.addEventListener("blur", () => log("blur"));
  document.addEventListener("visibilitychange", () => log("visibilitychange", { state: document.visibilityState }));
  window.addEventListener("pagehide", () => { log("page_unload"); flush(); });
  window.addEventListener("beforeunload", () => { log("page_unload"); flush(); });
  setInterval(flush, 3000);
})();
"""


def _css(spec: WebsiteSpec) -> str:
    return f"""
    :root {{
      --accent: {spec.accent};
      --ink: #102331;
      --muted: #617283;
      --paper: #fff8ea;
      --card: rgba(255, 251, 242, 0.86);
      --line: rgba(16, 35, 49, 0.12);
      --shadow: 0 24px 70px rgba(17, 44, 61, 0.16);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Aptos Display", "Trebuchet MS", "Gill Sans", sans-serif;
      background:
        radial-gradient(circle at 12% 10%, rgba(20, 184, 166, 0.20), transparent 30rem),
        radial-gradient(circle at 88% 8%, rgba(240, 173, 53, 0.18), transparent 26rem),
        linear-gradient(135deg, #fff8ea 0%, #eef8f4 54%, #f7e4d4 100%);
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .22;
      background-image:
        linear-gradient(rgba(16, 35, 49, .08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(16, 35, 49, .08) 1px, transparent 1px);
      background-size: 52px 52px;
      mask-image: linear-gradient(to bottom, black, transparent 82%);
    }}
    .ambient-layer {{
      position: fixed;
      inset: 0;
      pointer-events: none;
      overflow: hidden;
      z-index: -1;
    }}
    .ambient-layer i {{
      position: absolute;
      width: 260px;
      height: 260px;
      border-radius: 42%;
      filter: blur(6px);
      opacity: .25;
      background: linear-gradient(135deg, var(--accent), #f0ad35);
      animation: drift 18s ease-in-out infinite alternate;
    }}
    .ambient-layer i:nth-child(1) {{ left: -60px; top: 28%; }}
    .ambient-layer i:nth-child(2) {{ right: 10%; top: 18%; animation-delay: -5s; transform: scale(.72) rotate(18deg); }}
    .ambient-layer i:nth-child(3) {{ right: -80px; bottom: 4%; animation-delay: -9s; transform: scale(1.12) rotate(-12deg); }}
    a {{ color: inherit; }}
    .site-nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 18px;
      width: min(1220px, calc(100% - 28px));
      margin: 14px auto 0;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 251, 242, 0.86);
      box-shadow: 0 14px 45px rgba(17, 44, 61, 0.10);
      backdrop-filter: blur(18px);
    }}
    .brand {{ display: inline-flex; align-items: center; gap: 10px; text-decoration: none; font-weight: 950; min-width: max-content; }}
    .brand-mark {{ width: 28px; height: 28px; border-radius: 10px; background: linear-gradient(135deg, var(--accent), #f0ad35); display: inline-block; }}
    .site-nav nav {{ display: flex; gap: 8px; flex-wrap: wrap; flex: 1; }}
    .site-nav nav a {{ text-decoration: none; padding: 9px 11px; border-radius: 999px; color: var(--muted); font-size: 14px; font-weight: 800; }}
    .site-nav nav a:hover {{ color: var(--ink); background: rgba(255,255,255,0.72); }}
    .graph-pill {{ max-width: 260px; padding: 8px 10px; border-radius: 999px; color: #fff; background: var(--accent); font-size: 12px; font-weight: 900; text-align: center; }}
    main {{ width: min(1220px, calc(100% - 28px)); margin: 0 auto; padding: 28px 0 64px; }}
    h1 {{ margin: 10px 0 12px; font-size: clamp(44px, 8vw, 84px); line-height: .9; letter-spacing: -.07em; }}
    h2 {{ margin: 0 0 12px; font-size: 28px; letter-spacing: -.04em; }}
    p {{ color: var(--muted); font-size: 18px; line-height: 1.55; }}
    .kicker {{ color: var(--accent); font-size: 12px; font-weight: 950; letter-spacing: .18em; text-transform: uppercase; margin: 0; }}
    .panel, .commerce-hero, .docs-hero, .news-hero, .support-hero, .transition-panel, .docs-layout, .article-shell, .support-article, .product-card, .doc-card, .story-card, .support-card {{
      border: 1px solid var(--line);
      background: var(--card);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }}
    .commerce-hero, .docs-hero, .news-hero, .support-hero {{
      min-height: 360px;
      border-radius: 38px;
      padding: 36px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 26px;
      align-items: stretch;
      overflow: hidden;
    }}
    .compact {{ min-height: 260px; }}
    .button {{ display: inline-flex; align-items: center; justify-content: center; padding: 13px 16px; border-radius: 16px; color: #fff; text-decoration: none; background: linear-gradient(135deg, var(--accent), #112c3d); font-weight: 950; }}
    .button.ghost {{ color: var(--ink); background: rgba(255,255,255,.72); border: 1px solid var(--line); }}
    .search-box, .support-search {{ display: flex; justify-content: space-between; gap: 12px; margin-top: 22px; padding: 16px; border-radius: 18px; background: rgba(255,255,255,.70); border: 1px solid var(--line); color: var(--muted); }}
    .search-box a {{ color: var(--accent); font-weight: 950; text-decoration: none; }}
    .promo-card, .product-stage, .checkout-card, .terminal, .briefing, .status-stack, .contact-panel {{
      border-radius: 30px;
      padding: 24px;
      background: linear-gradient(135deg, rgba(255,255,255,.76), rgba(255,255,255,.36));
      border: 1px solid rgba(255,255,255,.8);
    }}
    .promo-card, .product-stage {{ display: grid; place-items: center; text-align: center; }}
    .promo-card strong, .checkout-card span {{ display: block; font-size: 54px; letter-spacing: -.06em; margin: 8px 0; }}
    .product-orb {{ width: 180px; height: 180px; display: grid; place-items: center; border-radius: 48px; color: #fff; background: linear-gradient(135deg, var(--accent), #f0ad35); font-size: 54px; font-weight: 950; transform: rotate(7deg); }}
    .rating {{ color: var(--muted); margin-top: 20px; font-weight: 800; }}
    .price-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .price-row strong {{ font-size: 54px; letter-spacing: -.06em; margin-right: 8px; }}
    .cards, .doc-grid, .story-grid, .support-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-top: 18px; }}
    .product-card, .doc-card, .story-card, .support-card {{ display: grid; gap: 10px; min-height: 210px; border-radius: 28px; padding: 18px; text-decoration: none; color: var(--ink); }}
    .product-card, .doc-card, .story-card, .support-card, .link-card, .topic-chip, .button {{
      transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
    }}
    .product-card:hover, .doc-card:hover, .story-card:hover, .support-card:hover, .link-card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 24px 60px rgba(17, 44, 61, .16);
    }}
    .product-image {{ width: 76px; height: 76px; border-radius: 24px; color: #fff; display: grid; place-items: center; background: linear-gradient(135deg, var(--accent), #f0ad35); font-weight: 950; }}
    .product-card b {{ font-size: 28px; letter-spacing: -.04em; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }}
    .panel {{ border-radius: 30px; padding: 24px; }}
    .spec-list {{ color: var(--muted); line-height: 1.9; padding-left: 20px; }}
    .docs-layout, .article-shell, .support-article {{ border-radius: 34px; padding: 22px; display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 22px; }}
    .docs-sidebar, .related {{ display: grid; align-content: start; gap: 10px; color: var(--muted); }}
    .docs-sidebar a, .related a {{ text-decoration: none; padding: 10px 12px; border-radius: 14px; background: rgba(255,255,255,.58); }}
    .docs-article, .news-article {{ min-width: 0; }}
    pre {{ overflow: auto; padding: 18px; border-radius: 18px; color: #e6f7ff; background: #102331; }}
    .terminal {{ color: #d9f99d; background: #102331; font-family: "Cascadia Mono", Consolas, monospace; display: grid; gap: 12px; align-content: center; }}
    .briefing, .status-stack {{ display: grid; gap: 12px; align-content: center; }}
    .topic-chip {{ display: inline-flex; width: fit-content; padding: 8px 11px; border-radius: 999px; color: #fff; background: var(--accent); text-decoration: none; font-weight: 900; }}
    .story-card span {{ color: var(--accent); font-weight: 950; text-transform: uppercase; letter-spacing: .12em; font-size: 12px; }}
    .story-card strong {{ font-size: 24px; letter-spacing: -.04em; line-height: 1.05; }}
    .news-article h1 {{ font-size: clamp(42px, 7vw, 72px); }}
    .byline {{ font-size: 14px; font-weight: 900; color: var(--accent); }}
    blockquote {{ margin: 24px 0; padding: 18px 22px; border-left: 6px solid var(--accent); background: rgba(255,255,255,.62); border-radius: 0 18px 18px 0; font-size: 22px; line-height: 1.35; }}
    .support-card span {{ width: 52px; height: 52px; display: grid; place-items: center; border-radius: 18px; color: #fff; background: var(--accent); font-size: 24px; font-weight: 950; }}
    .status-stack span {{ padding: 14px; border-radius: 16px; background: rgba(255,255,255,.62); font-weight: 900; }}
    input, textarea {{ width: 100%; border: 1px solid var(--line); border-radius: 14px; padding: 12px; margin: 10px 0; font: inherit; background: rgba(255,255,255,.78); }}
    .steps {{ color: var(--muted); line-height: 2; }}
    .transition-panel {{ margin-top: 18px; border-radius: 34px; padding: 24px; display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 22px; }}
    .experience-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .experience-strip article {{
      position: relative;
      overflow: hidden;
      min-height: 130px;
      border-radius: 28px;
      padding: 20px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, .58);
      box-shadow: 0 18px 48px rgba(17, 44, 61, .10);
    }}
    .experience-strip article::after {{
      content: "";
      position: absolute;
      right: -34px;
      top: -34px;
      width: 98px;
      height: 98px;
      border-radius: 32px;
      background: var(--accent);
      opacity: .16;
      transform: rotate(16deg);
    }}
    .experience-strip span {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 950;
      letter-spacing: .14em;
      text-transform: uppercase;
    }}
    .experience-strip strong {{
      display: block;
      margin: 12px 0 8px;
      font-size: clamp(26px, 3vw, 42px);
      line-height: .92;
      letter-spacing: -.06em;
    }}
    .link-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .link-card {{ display: grid; gap: 6px; text-decoration: none; padding: 15px; border-radius: 20px; color: var(--ink); background: rgba(255,255,255,.66); border: 1px solid var(--line); font-weight: 950; }}
    .link-card small, small {{ color: var(--muted); font-weight: 700; }}
    @media (max-width: 920px) {{
      .site-nav, .commerce-hero, .docs-hero, .news-hero, .support-hero, .docs-layout, .article-shell, .support-article, .transition-panel, .two-col, .experience-strip {{ grid-template-columns: 1fr; }}
      .graph-pill {{ display: none; }}
      .site-nav {{ align-items: flex-start; flex-direction: column; }}
    }}
    @keyframes drift {{
      from {{ transform: translate3d(0, 0, 0) rotate(0deg); }}
      to {{ transform: translate3d(28px, -24px, 0) rotate(22deg); }}
    }}
    """
