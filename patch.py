with open("tools/splatlens_viewer.html", "r") as f:
    content = f.read()

# Make sure spacebar action also updates btn visual classes accurately by focusing if needed, or by testing classList toggle directly
search = """    function updateBtns() {
      if (playing) {
        btnPlay.classList.add('active'); btnPause.classList.remove('active');
        btnPlay.setAttribute('aria-pressed', 'true'); btnPause.setAttribute('aria-pressed', 'false');
      } else {
        btnPlay.classList.remove('active'); btnPause.classList.add('active');
        btnPlay.setAttribute('aria-pressed', 'false'); btnPause.setAttribute('aria-pressed', 'true');
      }
      stepLabel.textContent = visibleSteps + '/' + T.length; scrubber.value = Math.max(0, visibleSteps - 1);
    }"""

replace = """    function updateBtns() {
      btnPlay.classList.toggle('active', playing); btnPause.classList.toggle('active', !playing);
      btnPlay.setAttribute('aria-pressed', playing); btnPause.setAttribute('aria-pressed', !playing);
      stepLabel.textContent = visibleSteps + '/' + T.length; scrubber.value = Math.max(0, visibleSteps - 1);
    }"""

content = content.replace(search, replace, 1)

with open("tools/splatlens_viewer.html", "w") as f:
    f.write(content)
