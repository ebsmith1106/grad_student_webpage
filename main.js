// EB Smith portfolio: rolling image strip, scroll reveals, header behavior.
(function () {
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- Rolling image strip: duplicate the items once so the loop is seamless ----
  document.querySelectorAll('.marquee-track').forEach(function (track) {
    if (reduceMotion) return;
    Array.prototype.slice.call(track.children).forEach(function (item) {
      var clone = item.cloneNode(true);
      clone.setAttribute('aria-hidden', 'true');
      clone.querySelectorAll('a').forEach(function (a) { a.tabIndex = -1; });
      track.appendChild(clone);
    });
    // Keep a steady speed (~45 px/s) whatever the screen width or number of photos
    function setSpeed() {
      track.style.setProperty('--roll-duration', (track.scrollWidth / 2 / 45) + 's');
    }
    window.addEventListener('load', setSpeed);
    window.addEventListener('resize', setSpeed);
    setSpeed();
  });

  // ---- Fade-in on scroll ----
  var reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && !reduceMotion) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add('in'); });
  }

  // ---- Header: border after scrolling, hide on scroll down, show on scroll up ----
  var header = document.querySelector('.site-header');
  var nav = document.getElementById('site-nav');
  var lastY = window.scrollY;
  window.addEventListener('scroll', function () {
    var y = window.scrollY;
    header.classList.toggle('scrolled', y > 10);
    var menuOpen = nav && nav.classList.contains('open');
    header.classList.toggle('hide', y > lastY && y > 300 && !menuOpen);
    lastY = y;
  }, { passive: true });

  // ---- Mobile menu ----
  var btn = document.querySelector('.menu-btn');
  if (btn && nav) {
    btn.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      btn.setAttribute('aria-expanded', open);
      btn.textContent = open ? 'Close' : 'Menu';
    });
    nav.querySelectorAll('a').forEach(function (a) {
      a.addEventListener('click', function () {
        nav.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
        btn.textContent = 'Menu';
      });
    });
  }

  // ---- Highlight the nav link for the section in view ----
  var links = nav ? nav.querySelectorAll('a[href^="#"]') : [];
  if (links.length && 'IntersectionObserver' in window) {
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) {
          a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id);
        });
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    links.forEach(function (a) {
      var s = document.querySelector(a.getAttribute('href'));
      if (s) spy.observe(s);
    });
  }
})();
