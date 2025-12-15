// Timeline configuration
const config = {
  margin: { top: 60, right: 60, bottom: 60, left: 60 },
  branchSpacing: 30,
  circleRadius: 8,
  maxCircleRadius: 20,
};

// Emotion color palette
const emotionColors = {
  sadness: "#4A90E2",
  joy: "#7ED321",
  anger: "#D0021B",
  fear: "#9013FE",
  disgust: "#F5A623",
  surprise: "#F8E71C",
  neutral: "#9B9B9B",
};

// Main initialization
async function init() {
  // Load data
  const rawData = await d3.json("data/timeline_data.json");

  // CRITICAL: Convert years to numbers and clean data
  const data = rawData
    .map((d) => ({
      ...d,
      year: +d.year, // Force to number
      num_chunks: +d.num_chunks,
      sadness: +d.sadness,
      joy: +d.joy,
      fear: +d.fear,
      anger: +d.anger,
      disgust: +d.disgust,
      surprise: +d.surprise,
      neutral: +d.neutral,
      emotion_purity: +d.emotion_purity,
    }))
    .filter((d) => !isNaN(d.year) && d.year > 0); // Remove invalid years

  console.log("Cleaned data:", data.length, "tales");
  console.log("First tale:", data[0]);
  console.log(
    "Year range:",
    d3.extent(data, (d) => d.year)
  );

  // Set up SVG dimensions
  const container = d3.select("#timeline-container");
  const containerWidth = container.node().getBoundingClientRect().width;

  const width = containerWidth - config.margin.left - config.margin.right;
  const height = 600 - config.margin.top - config.margin.bottom;
  const timelineY = height / 2;

  const svg = d3
    .select("#timeline")
    .attr("width", containerWidth)
    .attr("height", 600);

  const g = svg
    .append("g")
    .attr(
      "transform",
      `translate(${config.margin.left}, ${config.margin.top})`
    );

  // Create scales
  const xScale = d3
    .scaleLinear()
    // .domain(d3.extent(data, (d) => d.year))
    .domain([1805, 1875])
    .range([0, width])
    .nice();

  const sizeScale = d3
    .scaleSqrt()
    .domain(d3.extent(data, (d) => d.num_chunks))
    .range([config.circleRadius, config.maxCircleRadius]);

  console.log("xScale domain:", xScale.domain());
  console.log("xScale range:", xScale.range());

  // Test the scale
  const testYear = data[0].year;
  const testX = xScale(testYear);
  console.log(`Test: year ${testYear} → x position ${testX}`);

  // Draw main timeline
  g.append("line")
    .attr("class", "timeline-line")
    .attr("x1", 0)
    .attr("x2", width)
    .attr("y1", timelineY)
    .attr("y2", timelineY);

  // Group tales by year - ROUND dates to handle any decimals
  const talesByYear = d3.group(data, (d) => Math.round(d.year));

  console.log("Grouped by year:", Array.from(talesByYear.keys()));

  // Calculate positions for each tale
  const talesWithPositions = [];
  talesByYear.forEach((tales, year) => {
    const numTales = tales.length;
    console.log(`Year ${year}: ${numTales} tales`);

    tales.forEach((tale, index) => {
      const offsetIndex = index - (numTales - 1) / 2;
      const branchY = timelineY + offsetIndex * config.branchSpacing;
      const x = xScale(year);

      // Add calculated positions to tale object
      tale.branchY = branchY;
      tale.x = x;

      if (index === 0) {
        console.log(
          `  First tale "${tale.tale}": year=${year}, x=${x}, branchY=${branchY}`
        );
      }

      talesWithPositions.push(tale);
    });
  });

  console.log("Total positioned tales:", talesWithPositions.length);

  // Draw branch lines
  g.selectAll(".branch-line")
    .data(talesWithPositions)
    .join("line")
    .attr("class", "branch-line")
    .attr("x1", (d) => d.x)
    .attr("y1", timelineY)
    .attr("x2", (d) => d.x)
    .attr("y2", (d) => d.branchY)
    .style("opacity", (d) => (Math.abs(d.branchY - timelineY) > 5 ? 0.6 : 0));

  // Create tooltip
  const tooltip = d3.select("#tooltip");

  // Draw tale groups
  const taleGroups = g
    .selectAll(".tale-group")
    .data(talesWithPositions)
    .join("g")
    .attr("class", "tale-group")
    .attr("transform", (d) => `translate(${d.x}, ${d.branchY})`);

  // Draw circles or diamonds
  taleGroups.each(function (d) {
    const group = d3.select(this);
    const size = sizeScale(d.num_chunks);
    const color = emotionColors[d.dominant_emotion] || "#999";
    const opacity = calculateOpacity(d.emotion_purity);

    if (d.approximate) {
      // Diamond for approximate years
      const points = [
        [0, -size],
        [size, 0],
        [0, size],
        [-size, 0],
      ];

      group
        .append("polygon")
        .attr("class", "tale-shape")
        .attr("points", points.map((p) => p.join(",")).join(" "))
        .style("fill", color)
        .style("opacity", opacity)
        .style("stroke", "none");
    } else {
      // Circle for certain years
      group
        .append("circle")
        .attr("class", "tale-shape")
        .attr("r", size)
        .style("fill", color)
        .style("opacity", opacity)
        .style("stroke", "none");
    }
  });

  // Add interactions
  taleGroups
    .on("mouseover", function (event, d) {
      d3.select(this)
        .select(".tale-shape")
        .style("stroke", "#ffffff")
        .style("stroke-width", 3);

      showTooltip(event, d, tooltip);
    })
    .on("mouseout", function () {
      d3.select(this).select(".tale-shape").style("stroke", "none");

      tooltip.style("opacity", 0);
    })
    .on("click", (event, d) => {
      console.log("Clicked:", d.tale);
    });

  // Draw axis
  const xAxis = d3.axisBottom(xScale).tickFormat(d3.format("d")).ticks(10);

  g.append("g")
    .attr("transform", `translate(0, ${timelineY})`)
    .call(xAxis)
    .call((g) => g.select(".domain").attr("stroke", "#666"))
    .call((g) => g.selectAll(".tick line").attr("stroke", "#666"))
    .call((g) =>
      g.selectAll(".tick text").attr("fill", "#aaaaaa").attr("y", 10)
    );

  // Draw legend
  drawLegend();

  console.log("Visualization complete!");
}

// Calculate opacity based on emotion purity
function calculateOpacity(purity) {
  if (isNaN(purity) || purity === null || purity === undefined) {
    return 0.5;
  }
  return Math.max(0.3, Math.min(1.0, 0.3 + purity * 1.4));
}

// Show tooltip
function showTooltip(event, d, tooltip) {
  const emotions = [
    { name: "Sadness", value: d.sadness, color: emotionColors.sadness },
    { name: "Joy", value: d.joy, color: emotionColors.joy },
    { name: "Fear", value: d.fear, color: emotionColors.fear },
    { name: "Anger", value: d.anger, color: emotionColors.anger },
    { name: "Disgust", value: d.disgust, color: emotionColors.disgust },
    { name: "Surprise", value: d.surprise, color: emotionColors.surprise },
    { name: "Neutral", value: d.neutral, color: emotionColors.neutral },
  ].sort((a, b) => b.value - a.value);

  const emotionBars = emotions
    .map(
      (e) => `
    <div class="tooltip-emotion-bar">
      <div class="tooltip-emotion-label">${e.name}</div>
      <div class="tooltip-emotion-value">
        <div class="tooltip-emotion-fill" style="width: ${
          e.value * 100
        }%; background-color: ${e.color};"></div>
      </div>
      <div class="tooltip-emotion-percent">${(e.value * 100).toFixed(0)}%</div>
    </div>
  `
    )
    .join("");

  const content = `
    <div class="tooltip-title">${d.tale}</div>
    <div class="tooltip-content">
      <div style="margin-bottom: 8px;">
        <strong>Published:</strong> ${d.year}${
    d.approximate ? " (approximate)" : ""
  }<br>
        <strong>Length:</strong> ${d.num_chunks} chunks
      </div>
      <div style="margin-bottom: 4px;"><strong>Emotional Profile:</strong></div>
      ${emotionBars}
    </div>
  `;

  tooltip
    .html(content)
    .style("left", event.pageX + 15 + "px")
    .style("top", event.pageY - 15 + "px")
    .style("opacity", 1);
}

// Draw legend
function drawLegend() {
  const legend = d3.select("#legend");

  Object.entries(emotionColors).forEach(([emotion, color]) => {
    const item = legend.append("div").attr("class", "legend-item");

    item
      .append("div")
      .attr("class", "legend-circle")
      .style("background-color", color);

    item
      .append("span")
      .attr("class", "legend-label")
      .text(emotion.charAt(0).toUpperCase() + emotion.slice(1));
  });

  // Add this function to your timeline.js
  function addBiographicalMarkers(g, xScale, timelineY) {
    const lifeEvents = [
      { year: 1805, label: "Born in Odense", type: "birth", color: "#7ED321" },
      {
        year: 1819,
        label: "Moves to Copenhagen",
        type: "life",
        color: "#4A90E2",
      },
      {
        year: 1835,
        label: "First fairy tales published",
        type: "career",
        color: "#F8E71C",
      },
      { year: 1838, label: "Mother dies", type: "personal", color: "#D0021B" },
      {
        year: 1847,
        label: "Travels to England",
        type: "travel",
        color: "#9B9B9B",
      },
      {
        year: 1867,
        label: "Honorary citizen of Odense",
        type: "honor",
        color: "#9013FE",
      },
      {
        year: 1872,
        label: "Last tale published",
        type: "career",
        color: "#F8E71C",
      },
      {
        year: 1875,
        label: "Dies in Copenhagen",
        type: "death",
        color: "#D0021B",
      },
    ];

    // Add background shading for periods
    const periods = [
      {
        start: 1805,
        end: 1819,
        label: "Childhood",
        color: "rgba(126, 211, 33, 0.05)",
      },
      {
        start: 1819,
        end: 1835,
        label: "Education & Early Writing",
        color: "rgba(74, 144, 226, 0.05)",
      },
      {
        start: 1835,
        end: 1872,
        label: "Fairy Tale Period",
        color: "rgba(248, 231, 28, 0.08)",
      },
      {
        start: 1872,
        end: 1875,
        label: "Final Years",
        color: "rgba(208, 2, 27, 0.05)",
      },
    ];

    // Draw period backgrounds
    periods.forEach((period) => {
      g.append("rect")
        .attr("x", xScale(period.start))
        .attr("y", 0)
        .attr("width", xScale(period.end) - xScale(period.start))
        .attr("height", timelineY * 2)
        .attr("fill", period.color)
        .attr("pointer-events", "none");

      // Period label at top
      g.append("text")
        .attr("x", (xScale(period.start) + xScale(period.end)) / 2)
        .attr("y", -10)
        .attr("text-anchor", "middle")
        .attr("fill", "#666")
        .attr("font-size", "11px")
        .attr("font-style", "italic")
        .text(period.label);
    });

    // Draw life events
    lifeEvents.forEach((event) => {
      const x = xScale(event.year);
      const markerY =
        event.type === "birth" || event.type === "death"
          ? timelineY
          : timelineY + (event.year % 2 === 0 ? -80 : 80);

      // Marker line
      g.append("line")
        .attr("x1", x)
        .attr("y1", timelineY - 5)
        .attr("x2", x)
        .attr("y2", markerY)
        .attr("stroke", event.color)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "3,3")
        .attr("opacity", 0.6);

      // Marker dot
      g.append("circle")
        .attr("cx", x)
        .attr("cy", markerY)
        .attr("r", 5)
        .attr("fill", event.color)
        .attr("stroke", "#1a1a1a")
        .attr("stroke-width", 2);

      // Label
      g.append("text")
        .attr("x", x)
        .attr("y", markerY + (markerY > timelineY ? 20 : -15))
        .attr("text-anchor", "middle")
        .attr("fill", "#aaaaaa")
        .attr("font-size", "10px")
        .text(event.label)
        .style("pointer-events", "none");
    });
  }

  // Call it in your init() function after drawing the timeline:
  addBiographicalMarkers(g, xScale, timelineY);

  // Add legend for approximate years
  const shapeItem = legend
    .append("div")
    .attr("class", "legend-item")
    .style("margin-left", "20px");

  shapeItem
    .append("div")
    .style("width", "16px")
    .style("height", "16px")
    .style("background-color", "#666")
    .style("transform", "rotate(45deg)");

  shapeItem
    .append("span")
    .attr("class", "legend-label")
    .text("Approximate year");
}

// Initialize
init();
