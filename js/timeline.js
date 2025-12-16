// Timeline configuration
const config = {
  margin: { top: 100, right: 60, bottom: 80, left: 60 },
  verticalSpacing: 25, // Space between stacked tales
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

// Sort order for emotions (for visual grouping)
const emotionOrder = {
  joy: 1,
  surprise: 2,
  neutral: 3,
  sadness: 4,
  fear: 5,
  anger: 6,
  disgust: 7,
};

async function init() {
  // Load data
  const rawData = await d3.json("data/timeline_data.json");

  const data = rawData
    .map((d) => ({
      ...d,
      year: +d.year,
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
    .filter((d) => !isNaN(d.year) && d.year > 0);

  console.log("Cleaned data:", data.length, "tales");

  // Set up SVG dimensions
  const container = d3.select("#timeline-container");
  const containerWidth = container.node().getBoundingClientRect().width;

  const width = containerWidth - config.margin.left - config.margin.right;
  const height = 800 - config.margin.top - config.margin.bottom; // Taller for upward stacking
  const timelineY = height; // TIMELINE AT BOTTOM

  const svg = d3
    .select("#timeline")
    .attr("width", containerWidth)
    .attr("height", 800);

  const g = svg
    .append("g")
    .attr(
      "transform",
      `translate(${config.margin.left}, ${config.margin.top})`
    );

  // Create scales
  const xScale = d3.scaleLinear().domain([1805, 1875]).range([0, width]);

  const sizeScale = d3
    .scaleSqrt()
    .domain(d3.extent(data, (d) => d.num_chunks))
    .range([config.circleRadius, config.maxCircleRadius]);

  // Add biographical context
  addBiographicalContext(g, xScale, timelineY, width, height);

  // Draw main timeline at bottom
  g.append("line")
    .attr("class", "timeline-line")
    .attr("x1", 0)
    .attr("x2", width)
    .attr("y1", timelineY)
    .attr("y2", timelineY)
    .attr("stroke", "#666")
    .attr("stroke-width", 3);

  // Group tales by year
  const talesByYear = d3.group(data, (d) => Math.round(d.year));

  // Calculate positions with smart sorting
  const talesWithPositions = [];

  talesByYear.forEach((tales, year) => {
    // SORT TALES WITHIN EACH YEAR
    // Priority: 1) Certain dates first, 2) By emotion, 3) By size (small to large)
    const sortedTales = tales.sort((a, b) => {
      // 1. Confirmed dates before approximate
      if (a.approximate !== b.approximate) {
        return a.approximate ? 1 : -1;
      }

      // 2. Group by emotion (creates color bands)
      const emotionDiff =
        emotionOrder[a.dominant_emotion] - emotionOrder[b.dominant_emotion];
      if (emotionDiff !== 0) return emotionDiff;

      // 3. Smaller tales first (creates pyramid effect)
      return a.num_chunks - b.num_chunks;
    });

    // Calculate vertical positions (stack upward from timeline)
    sortedTales.forEach((tale, index) => {
      const taleSize = sizeScale(tale.num_chunks);

      // Stack upward: first tale just above timeline, each subsequent tale stacks higher
      // Add extra spacing based on previous tale's size
      let cumulativeHeight = 0;
      for (let i = 0; i < index; i++) {
        const prevSize = sizeScale(sortedTales[i].num_chunks);
        cumulativeHeight += prevSize * 2 + config.verticalSpacing;
      }

      tale.y = timelineY - (taleSize + cumulativeHeight + 10); // Stack upward
      tale.x = xScale(year);
      tale.size = taleSize;

      talesWithPositions.push(tale);
    });
  });

  // Find max height for scaling if needed
  const maxHeight = Math.abs(d3.min(talesWithPositions, (d) => d.y));
  console.log("Max stack height:", maxHeight);

  // Draw connecting lines (from timeline up to tale)
  g.selectAll(".stack-line")
    .data(talesWithPositions)
    .join("line")
    .attr("class", "stack-line")
    .attr("x1", (d) => d.x)
    .attr("y1", timelineY)
    .attr("x2", (d) => d.x)
    .attr("y2", (d) => d.y)
    .attr("stroke", "#444")
    .attr("stroke-width", 1)
    .attr("opacity", 0.2);

  // Create tooltip
  const tooltip = d3.select("#tooltip");

  // Draw tale groups
  const taleGroups = g
    .selectAll(".tale-group")
    .data(talesWithPositions)
    .join("g")
    .attr("class", "tale-group")
    .attr("transform", (d) => `translate(${d.x}, ${d.y})`);

  // Draw shapes
  taleGroups.each(function (d) {
    const group = d3.select(this);
    const color = emotionColors[d.dominant_emotion] || "#999";
    const opacity = calculateOpacity(d.emotion_purity);

    if (d.approximate) {
      // Diamond for approximate
      const points = [
        [0, -d.size],
        [d.size, 0],
        [0, d.size],
        [-d.size, 0],
      ];

      group
        .append("polygon")
        .attr("class", "tale-shape")
        .attr("points", points.map((p) => p.join(",")).join(" "))
        .style("fill", color)
        .style("opacity", opacity)
        .style("stroke", "none");
    } else {
      // Circle for confirmed
      group
        .append("circle")
        .attr("class", "tale-shape")
        .attr("r", d.size)
        .style("fill", color)
        .style("opacity", opacity)
        .style("stroke", "none");
    }
  });

  // Interactions
  taleGroups
    .on("mouseover", function (event, d) {
      // Highlight this tale
      d3.select(this)
        .select(".tale-shape")
        .style("stroke", "#ffffff")
        .style("stroke-width", 3);

      // Dim other tales
      taleGroups.style("opacity", 0.3);
      d3.select(this).style("opacity", 1);

      // Highlight the connecting line
      g.selectAll(".stack-line").attr("opacity", (line) =>
        line.tale === d.tale ? 0.6 : 0.1
      );

      showTooltip(event, d, tooltip);
    })
    .on("mouseout", function () {
      d3.select(this).select(".tale-shape").style("stroke", "none");

      taleGroups.style("opacity", 1);
      g.selectAll(".stack-line").attr("opacity", 0.2);

      tooltip.style("opacity", 0);
    })
    .on("click", (event, d) => {
      console.log("Clicked:", d.tale);
    });

  // Draw axis at bottom
  const xAxis = d3
    .axisBottom(xScale)
    .tickFormat(d3.format("d"))
    .tickValues([1805, 1820, 1835, 1850, 1865, 1875])
    .tickSize(10);

  g.append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0, ${timelineY})`)
    .call(xAxis)
    .call((g) => g.select(".domain").attr("stroke", "#666"))
    .call((g) => g.selectAll(".tick line").attr("stroke", "#666"))
    .call((g) =>
      g.selectAll(".tick text").attr("fill", "#aaaaaa").attr("y", 15)
    );

  // Draw legend
  drawLegend();

  console.log("Visualization complete!");
}

function createEmotionalFlowChart() {
  // Group tales by year and emotion
  const emotionsByYear = d3.group(data, (d) => d.year);

  const flowData = Array.from(emotionsByYear, ([year, tales]) => {
    const emotionCounts = {
      year: year,
      sadness: tales.filter((t) => t.dominant_emotion === "sadness").length,
      joy: tales.filter((t) => t.dominant_emotion === "joy").length,
      fear: tales.filter((t) => t.dominant_emotion === "fear").length,
      anger: tales.filter((t) => t.dominant_emotion === "anger").length,
      disgust: tales.filter((t) => t.dominant_emotion === "disgust").length,
      surprise: tales.filter((t) => t.dominant_emotion === "surprise").length,
      neutral: tales.filter((t) => t.dominant_emotion === "neutral").length,
    };
    return emotionCounts;
  });

  // Create stacked area chart
  const stack = d3
    .stack()
    .keys([
      "sadness",
      "joy",
      "fear",
      "anger",
      "disgust",
      "surprise",
      "neutral",
    ]);

  const series = stack(flowData);

  // Draw stacked area chart with your emotion colors
  const area = d3
    .area()
    .x((d) => xScale(d.data.year))
    .y0((d) => flowYScale(d[0]))
    .y1((d) => flowYScale(d[1]))
    .curve(d3.curveMonotoneX);

  flowG
    .selectAll(".emotion-layer")
    .data(series)
    .join("path")
    .attr("class", "emotion-layer")
    .attr("d", area)
    .style("fill", (d) => emotionColors[d.key])
    .style("opacity", 0.8);
}

// Biographical context
function addBiographicalContext(g, xScale, timelineY, width, height) {
  // Period backgrounds
  const periods = [
    {
      start: 1805,
      end: 1819,
      label: "Childhood",
      color: "rgba(126, 211, 33, 0.03)",
    },
    {
      start: 1819,
      end: 1835,
      label: "Education & Early Writing",
      color: "rgba(74, 144, 226, 0.03)",
    },
    {
      start: 1835,
      end: 1872,
      label: "Fairy Tale Period",
      color: "rgba(248, 231, 28, 0.05)",
    },
    {
      start: 1872,
      end: 1875,
      label: "Final Years",
      color: "rgba(208, 2, 27, 0.03)",
    },
  ];

  periods.forEach((period) => {
    g.append("rect")
      .attr("x", xScale(period.start))
      .attr("y", 0)
      .attr("width", xScale(period.end) - xScale(period.start))
      .attr("height", height)
      .attr("fill", period.color)
      .attr("pointer-events", "none");

    // Period label at top
    g.append("text")
      .attr("x", (xScale(period.start) + xScale(period.end)) / 2)
      .attr("y", -15)
      .attr("text-anchor", "middle")
      .attr("fill", "#666")
      .attr("font-size", "11px")
      .attr("font-style", "italic")
      .text(period.label);
  });

  // Life events
  const lifeEvents = [
    { year: 1805, label: "Born", color: "#7ED321" },
    { year: 1835, label: "First tales", color: "#F8E71C" },
    { year: 1872, label: "Last tale", color: "#F8E71C" },
    { year: 1875, label: "Died", color: "#D0021B" },
  ];

  lifeEvents.forEach((event) => {
    const x = xScale(event.year);

    // Marker at timeline
    g.append("circle")
      .attr("cx", x)
      .attr("cy", timelineY)
      .attr("r", 6)
      .attr("fill", event.color)
      .attr("stroke", "#1a1a1a")
      .attr("stroke-width", 2);

    // Label below timeline
    g.append("text")
      .attr("x", x)
      .attr("y", timelineY + 30)
      .attr("text-anchor", "middle")
      .attr("fill", "#aaaaaa")
      .attr("font-size", "10px")
      .text(event.label);
  });
}

function calculateOpacity(purity) {
  if (isNaN(purity) || purity === null || purity === undefined) {
    return 0.5;
  }
  return Math.max(0.3, Math.min(1.0, 0.3 + purity * 1.4));
}

function showTooltip(event, d, tooltip) {
  const emotions = [
    { name: "Sadness", value: d.sadness, color: emotionColors.sadness },
    { name: "Joy", value: d.joy, color: emotionColors.joy },
    // ... etc
  ].sort((a, b) => b.value - a.value);

  const emotionBars = emotions
    .map(
      (e) => `
      <div class="tooltip-emotion-bar">
        <div class="tooltip-emotion-label">${e.name}</div>
        <div class="tooltip-emotion-value">
          <div class="tooltip-emotion-fill" 
               style="width: ${e.value * 100}%; background-color: ${e.color};">
          </div>
        </div>
        <div class="tooltip-emotion-percent">${(e.value * 100).toFixed(
          0
        )}%</div>
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
        <strong>Length:</strong> ${d.num_chunks} chunks<br>
        <strong>Date Certainty:</strong> ${
          d.approximate ? "Approximate" : "Confirmed"
        }
      </div>
      
      <div style="margin-bottom: 4px;"><strong>Emotional Profile:</strong></div>
      ${emotionBars}
      
      <!-- ADD THESE NEW ELEMENTS -->
      <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #444;">
        <button onclick="exploreTale('${d.tale}')" 
                style="background: #4A90E2; color: white; border: none; 
                       padding: 6px 12px; border-radius: 4px; cursor: pointer; 
                       width: 100%;">
          Explore Tale Details →
        </button>
      </div>
    </div>
  `;

  tooltip
    .html(content)
    .style("left", event.pageX + 15 + "px")
    .style("top", event.pageY - 15 + "px")
    .style("opacity", 1);
}

// Add click handler to show more detail
function exploreTale(taleName) {
  // Option 1: Expand tooltip to show chunk-by-chunk breakdown
  // Option 2: Open modal with full emotional journey chart
  // Option 3: Navigate to dedicated tale detail page

  // I'd recommend modal approach:
  showTaleModal(taleName);
}

// In timeline.js
function initSearch() {
  const searchInput = d3.select("#tale-search");
  const clearButton = d3.select("#clear-search");
  const resultsDiv = d3.select("#search-results");

  searchInput.on("input", function () {
    const query = this.value.toLowerCase().trim();

    if (query.length === 0) {
      resetVisualization();
      resultsDiv.text("");
      return;
    }

    const matches = data.filter((d) => d.tale.toLowerCase().includes(query));

    if (matches.length === 0) {
      resultsDiv.text("No tales found");
      dimAllTales();
    } else {
      resultsDiv.text(`Found ${matches.length} tale(s)`);
      highlightTales(matches);
    }
  });

  clearButton.on("click", () => {
    searchInput.node().value = "";
    resetVisualization();
    resultsDiv.text("");
  });
}

function highlightTales(matches) {
  const matchTitles = new Set(matches.map((d) => d.tale));

  // Dim non-matches
  taleGroups.style("opacity", (d) => (matchTitles.has(d.tale) ? 1 : 0.2));

  // Optionally zoom to matches if they're in a narrow time range
  if (matches.length <= 5) {
    const years = matches.map((d) => d.year);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);

    // Animate to focus on this time period
    // (Implementation depends on your zoom setup)
  }
}

function dimAllTales() {
  taleGroups.style("opacity", 0.2);
}

function resetVisualization() {
  taleGroups.style("opacity", 1);
}

function drawLegend() {
  const legend = d3.select("#legend");
  legend.html(""); // Clear existing

  // Emotion colors
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

  // Shape legend
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
    .text("Approximate date");

  // Stacking note
  const noteItem = legend
    .append("div")
    .attr("class", "legend-item")
    .style("margin-left", "20px")
    .style("font-style", "italic")
    .style("color", "#888");

  noteItem.append("span").text("Tales stack by: certainty → emotion → size");
}

init();
