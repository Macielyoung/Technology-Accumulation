
var width = window.innerWidth;
var height = window.innerHeight;

var map = d3.select(".overview-map")
    .append("svg")
    .attr("class", "mapSvg")
    .attr("preserveAspectRatio", "xMidYMid meet")
    .attr("viewBox", "0 0 " + width + " " + height);

var colors = ["#cbe7ff", "#c0e2ff", "#abd9ff", "#92cfff", "#466b95"];

var mapGroup = map.append("g")
    .attr("class", "mapGroup");

var zoneGroup = map.append("g")
    .attr("class", "zoneGroup");

var proscale = width * 0.7;
var protran1 = -proscale - 50;
var protran2 = proscale + 100;
var projection = d3.geo.mercator()
    .scale(proscale)
    .translate([protran1, protran2]);

var path = d3.geo.path()
    .projection(projection);

d3.json("https://raw.githubusercontent.com/Macielyoung/Front-Learning/master/data/wholesale.json", function (error, valuedata) {
    //将读取到的数据存到数组values，令其索引号为各省的名称
    var values = {};

    for (var i = 0; i < valuedata.Province.length; i++) {
        var name = valuedata.Province[i].name;
        var value = valuedata.Province[i].value;
        values[name] = value;
    }

    var minv = d3.min(valuedata.Province, function (d) {
        return d.value;
    });
    console.log(minv);
    var maxv = d3.max(valuedata.Province, function (d) {
        return d.value;
    });

    var scale = d3.scale.quantize()
        .domain([minv, maxv])
        .range(colors);

    d3.json("https://raw.githubusercontent.com/Macielyoung/Front-Learning/master/data/china.json", function (error, root) {
        //添加中国各种的路径元素
        var provinces = mapGroup.selectAll("path")
            .data(root.features)
            .enter()
            .append("path")
            .attr("class", "mapPath")
            .attr("fill", function (d) {
                return scale(values[d.properties.name]);
            })
            .attr("d", path);
        console.log(provinces);
    });

    d3.json("https://raw.githubusercontent.com/Macielyoung/Front-Learning/master/data/location.json", function (error, places) {
        var zone = zoneGroup.selectAll(".zone")
            .data(places.Nation)
            .enter()
            .append("g")
            .attr("class", "zone")
            .attr("transform", function (d) {
                var coor = projection([d.log, d.lat]);
                return "translate(" + coor[0] + "," + coor[1] + ")";
            });

        var nation = zone.append("circle")
            .attr("r", 8)
            .attr("fill", "#1c5fa5")
            .attr("class", "nation")
            .style("stroke-width", 3)
            .style("stroke", "#ffffff");

        zone.on("mouseover", function (d) {
            // var mouse = d3.mouse(this);
            tooltip.html("Nation : " + d.name + "<br/> Click to See More")
                .style("left", d3.event.pageX + 20 + "px")
                .style("top", d3.event.pageY + 10 + "px")
                .style("opacity", 0.9);
        });

        zone.on("mouseout", function () {
            tooltip.style("left", "400px")
                .style("top", "50px")
                .style("opacity", 0);
        });
    });
});

var tooltip = d3.select(".overview-map")
    .append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

var zoom = d3.behavior.zoom()
    .translate([0, 0])
    .scale(1)
    .scaleExtent([1, 8])
    .on("zoom", zoomed);
map.call(zoom);

var saletip = d3.select(".overview-map")
    .append("div")
    .attr("class", "SaleTip")
    .style("height", "70px")
    .style("width", "181px")
    .html("Wholesale<br/>Turnover YTD");

//colorLegend module
var Legend = d3.select(".overview-map")
    .append("div")
    .attr("class", "Legend");

var colorLegend = Legend.append("svg").append("g")
    .attr("class", "colorLegend");

var rects = colorLegend.selectAll("rect")
    .data(colors)
    .enter()
    .append("rect")
    .attr("class", "colorRect")
    .attr("x", function (d, i) {
        return 36 * i;
    })
    .attr("fill", function (d) {
        return d;
    });

var low = colorLegend.append("text")
    .attr("class", "colorText")
    .attr("y", 40)
    .attr("x", 0)
    .text("低")
    .attr("fill", "#000");

var high = colorLegend.append("text")
    .attr("class", "colorText")
    .attr("y", 40)
    .attr("x", 160)
    .text("高")
    .attr("fill", "#000");

function zoomed() {
    var zoom_translate = d3.event.translate;
    var zoom_scale = d3.event.scale;
    mapGroup.attr("transform", "translate(" + zoom_translate + ")scale(" + zoom_scale + ")");
    zoneGroup.attr("transform", "translate(" + zoom_translate + ")scale(" + zoom_scale + ")");

    tooltip.style("left", "400px")
        .style("top", "50px")
        .style("opacity", 0);

    if (zoom_scale < 2.0) {
        zoneGroup.selectAll(".zone").remove();
        d3.json("https://raw.githubusercontent.com/Macielyoung/Front-Learning/master/data/location.json", function (error, places) {
            var zone = zoneGroup.selectAll(".zone")
                .data(places.Nation)
                .enter()
                .append("g")
                .attr("class", "zone")
                .attr("transform", function (d) {
                    var coor = projection([d.log, d.lat]);
                    return "translate(" + coor[0] + "," + coor[1] + ")";
                });

            var nation = zone.append("circle")
                .attr("r", 8)
                .attr("fill", "#1c5fa5")
                .attr("class", "nation")
                .style("stroke-width", 3)
                .style("stroke", "#ffffff");

            zone.on("mouseover", function (d) {
                // var mouse = d3.mouse(this);
                tooltip.html("Nation : " + d.name + "<br/> Click to See More")
                    .style("left", d3.event.pageX + 20 + "px")
                    .style("top", d3.event.pageY + 10 + "px")
                    .style("opacity", 0.9);
            });

            zone.on("mouseout", function () {
                tooltip.style("left", "400px")
                    .style("top", "50px")
                    .style("opacity", 0);
            });
        });
    }
    else {
        if (zoom_scale < 5.0 && zoom_scale > 3.0) {
            zoneGroup.selectAll(".zone").remove();
            d3.json("https://raw.githubusercontent.com/Macielyoung/Front-Learning/master/data/location.json", function (error, places) {
                var zone = zoneGroup.selectAll(".zone")
                    .data(places.Region)
                    .enter()
                    .append("g")
                    .attr("class", "zone")
                    .attr("transform", function (d) {
                        var coor = projection([d.log, d.lat]);
                        return "translate(" + coor[0] + "," + coor[1] + ")";
                    });

                var region = zone.append("circle")
                    .attr("r", 5)
                    .attr("fill", "#1c5fa5")
                    .attr("class", "region")
                    .style("stroke-width", 2)
                    .style("stroke", "#ffffff");

                zone.on("mouseover", function (d) {
                    // var mouse = d3.mouse(this);
                    tooltip.html("Region : " + d.name + "<br/> Click to See More")
                        .style("left", d3.event.pageX + 20 + "px")
                        .style("top", d3.event.pageY + 10 + "px")
                        .style("opacity", 0.9);
                });

                zone.on("mouseout", function () {
                    tooltip.style("left", "400px")
                        .style("top", "50px")
                        .style("opacity", 0);
                });
            });
        }
        else if (zoom_scale > 6.0) {
            zoneGroup.selectAll(".zone").remove();
            d3.json("https://raw.githubusercontent.com/Macielyoung/Front-Learning/master/data/location.json", function (error, places) {
                var zone = zoneGroup.selectAll(".zone")
                    .data(places.Province)
                    .enter()
                    .append("g")
                    .attr("class", "zone")
                    .attr("transform", function (d) {
                        var coor = projection([d.log, d.lat]);
                        return "translate(" + coor[0] + "," + coor[1] + ")";
                    });

                var province = zone.append("circle")
                    .attr("r", 2)
                    .attr("fill", "#1c5fa5")
                    .attr("class", "province")
                    .style("stroke-width", 1)
                    .style("stroke", "#ffffff");

                zone.on("mouseover", function (d) {
                    // var mouse = d3.mouse(this);
                    tooltip.html("Province : " + d.name + "<br/> Click to See More")
                        .style("left", d3.event.pageX + 20 + "px")
                        .style("top", d3.event.pageY + 10 + "px")
                        .style("opacity", 0.9);
                });

                zone.on("mouseout", function () {
                    tooltip.style("left", "400px")
                        .style("top", "50px")
                        .style("opacity", 0);
                });
            });
        }
    }
}