---
title: Diffusion modelling on fashion datasets 
mathjax: true
comments: true
tags:
  - Letter Boxed
  - algorithms
  - tree-search
  - games
  - diffusion
  - generative AI
categories: []
date: 2025-10-13 21:01:00
---

# Part 1 (Variational inference)

Let's say you have a continuous distribution $p(x)$ that is easy to sample from, but difficult to write in closed form. A common reason for this could be because the distribution is a complex joint distribution over many variables and marginalizing over multiple variables is intractable. What is a statistician to do here?

<p align="center">
  <img src="/2025/10/13/Introduction-to-denoising-diffusion-models-part-1/f_x_plot.png" alt="NYT Letter Boxed Main Page" style="max-width:500px height:auto; width:100%;">
</p>

One idea is that we can try to **approximate** $p(x)$ with a distribution that we know. 

#### Try approximating with a gaussian distribution

$$q(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

You can pick the mean $\mu$, and standard deviation $\sigma$ as you see fit

<div style="margin: 20px 0;">
  <div style="margin-bottom: 15px;">
    <div style="margin-bottom: 10px;">
      <strong>Mean ($\mu$):</strong>
      <div style="margin-top: 5px; margin-left: 10px;">
        <label style="margin-right: 15px;">
          <input type="radio" name="mean-select" value="-3.0" style="margin-right: 5px;">-3.0
        </label>
        <label style="margin-right: 15px;">
          <input type="radio" name="mean-select" value="0.0" checked style="margin-right: 5px;">0.0
        </label>
        <label style="margin-right: 15px;">
          <input type="radio" name="mean-select" value="4.0" style="margin-right: 5px;">4.0
        </label>
        <label style="margin-right: 15px;">
          <input type="radio" name="mean-select" value="7.0" style="margin-right: 5px;">7.0
        </label>
      </div>
    </div>
    <div>
      <strong>Standard Deviation ($\sigma$):</strong>
      <div style="margin-top: 5px;">
        <label style="margin-right: 15px;">
          <input type="radio" name="std-select" value="0.6" style="margin-right: 5px;">0.6
        </label>
        <label style="margin-right: 15px;">
          <input type="radio" name="std-select" value="0.8" style="margin-right: 5px;">0.8
        </label>
        <label style="margin-right: 15px;">
          <input type="radio" name="std-select" value="1.0" checked style="margin-right: 5px;">1.0
        </label>
        <label style="margin-right: 15px;">
          <input type="radio" name="std-select" value="3.0" style="margin-right: 5px;">3.0
        </label>
      </div>
    </div>
  </div>
  <div style="text-align: center;">
    <img id="approx-plot" src="/2025/10/13/Introduction-to-denoising-diffusion-models-part-1/approximations/p_x_approx_mean_-3.0_approx_std_0.6.png" alt="Gaussian Approximation" style="max-width:80%; height:auto; width:100%;">
  </div>
</div>

<script>
(function() {
  const meanRadios = document.getElementsByName('mean-select');
  const stdRadios = document.getElementsByName('std-select');
  const approxPlot = document.getElementById('approx-plot');
  
  function updatePlot() {
    let mean = '0.0';
    let std = '1.0';
    
    for (const radio of meanRadios) {
      if (radio.checked) {
        mean = radio.value;
        break;
      }
    }
    
    for (const radio of stdRadios) {
      if (radio.checked) {
        std = radio.value;
        break;
      }
    }
    
    const imagePath = `/2025/10/13/Introduction-to-denoising-diffusion-models-part-1/approximations/p_x_approx_mean_${mean}_approx_std_${std}.png`;
    approxPlot.src = imagePath;
  }
  
  for (const radio of meanRadios) {
    radio.addEventListener('change', updatePlot);
  }
  
  for (const radio of stdRadios) {
    radio.addEventListener('change', updatePlot);
  }
})();
</script>


