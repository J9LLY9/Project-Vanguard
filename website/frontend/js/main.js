(function () {
  "use strict";

  // Respect the user's OS-level motion preference — same rule as every
  // other animation in the hero. If set, skip this entirely: no listener,
  // no effect, nothing to opt back out of.
  var prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReducedMotion) return;

  var hero = document.querySelector(".hero");
  if (!hero) return;

  var GRAIN_MAX_OFFSET_PX = 5;  // deliberately tiny — felt more than seen
  var GRAIN_SMOOTHING = 0.08;   // fraction of remaining distance closed per frame
  var GLOW_SMOOTHING = 0.045;   // slower than grain on purpose — reads as
                                 // heavier/weightier; the speed difference
                                 // between layers is what creates the sense
                                 // of depth (see styles.css HERO — BACKGROUND)

  var GLOW_REACH = 0.4; // fraction of the cursor's distance from center that
                          // the glow actually travels — proportional to
                          // viewport size, so it behaves consistently from
                          // a phone to an ultrawide monitor

  var grainTargetX = 0, grainTargetY = 0;
  var grainCurrentX = 0, grainCurrentY = 0;

  var glowTargetX = 0, glowTargetY = 0;   // px offset from center — matches
  var glowCurrentX = 0, glowCurrentY = 0; // the CSS translate() fallback (0px)

  window.addEventListener("mousemove", function (event) {
    var nx = (event.clientX / window.innerWidth - 0.5) * 2;  // -1 .. 1
    var ny = (event.clientY / window.innerHeight - 0.5) * 2; // -1 .. 1
    grainTargetX = nx * GRAIN_MAX_OFFSET_PX;
    grainTargetY = ny * GRAIN_MAX_OFFSET_PX;

    glowTargetX = (event.clientX - window.innerWidth / 2) * GLOW_REACH;
    glowTargetY = (event.clientY - window.innerHeight / 2) * GLOW_REACH;
  });

  function tick() {
    // Exponential smoothing, evaluated every frame. A CSS `transition`
    // can't do this correctly here, because the target itself changes
    // every frame as the mouse moves — the transition just keeps getting
    // redirected before it ever catches up. Easing in JS, one small step
    // per frame, is the correct tool for a continuously-moving target.
    grainCurrentX += (grainTargetX - grainCurrentX) * GRAIN_SMOOTHING;
    grainCurrentY += (grainTargetY - grainCurrentY) * GRAIN_SMOOTHING;
    glowCurrentX += (glowTargetX - glowCurrentX) * GLOW_SMOOTHING;
    glowCurrentY += (glowTargetY - glowCurrentY) * GLOW_SMOOTHING;

    hero.style.setProperty("--grain-x", grainCurrentX.toFixed(2) + "px");
    hero.style.setProperty("--grain-y", grainCurrentY.toFixed(2) + "px");
    hero.style.setProperty("--glow-x", glowCurrentX.toFixed(2) + "px");
    hero.style.setProperty("--glow-y", glowCurrentY.toFixed(2) + "px");

    window.requestAnimationFrame(tick);
  }

  window.requestAnimationFrame(tick);
})();
