// ===================== YEAR SPAN (SHARED) =====================
var startYear = 2019;
var endYear   = 2024;
var startMonth = 4;   // April
var endMonth   = 11;  // November

// Derived date range + seasonal filter
var startDate = ee.Date.fromYMD(startYear, startMonth, 1);
var endDate   = ee.Date.fromYMD(endYear, endMonth, 1).advance(1, 'month').advance(-1, 'day');
var seasonalFilter = ee.Filter.calendarRange(startMonth, endMonth, 'month');


// ===================== LULC CONFIG (unchanged) =====================
var PROB_THRESHOLD = 0.0;
var EXPAND_MONTHS  = 1;
var USE_WORLDCOVER = true;

var DW_COLLECTION_ID = 'GOOGLE/DYNAMICWORLD/V1';
var WC_COLLECTION_ID = 'ESA/WorldCover/v200';

var DW_CLASS_NAMES = [
  'water','trees','grass','flooded_vegetation','crops',
  'shrub_and_scrub','built','bare','snow_and_ice'
];

var DW_PALETTE = [
  '419bdf','397d49','88b053','000000','e49635',
  'dfc35a','c4281b','a59b8f','b39fe1'
];


// ===================== AOI =====================
Map.centerObject(studyRegion, 12);
Map.addLayer(studyRegion, {color: 'blue'}, 'Study Region');


// ===================== HELPERS =====================
// (This mask keeps original S2 band names; no renaming)
function maskS2SR(img) {
  var scl = img.select('SCL');
  var good = scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  var qa60 = img.select('QA60');
  var cloudBit = 1 << 10, cirrusBit = 1 << 11;
  var qaMask = qa60.bitwiseAnd(cloudBit).eq(0).and(qa60.bitwiseAnd(cirrusBit).eq(0));
  return img.updateMask(good).updateMask(qaMask)
            .copyProperties(img, ['system:time_start']);
}

function m2(mm){ return (mm < 10 ? '0' + mm : '' + mm); }


// ===================== LULC (multi-year composite) =====================
function monthProb(geom, year, month){
  var start = ee.Date.fromYMD(year, month, 1);
  var end   = start.advance(1, 'month');
  return ee.ImageCollection(DW_COLLECTION_ID)
    .filterBounds(geom)
    .filterDate(start, end)
    .select(DW_CLASS_NAMES)
    .mean();
}

function monthProbExpanded(geom, year, month, expandN){
  var start = ee.Date.fromYMD(year, month, 1).advance(-expandN, 'month');
  var end   = ee.Date.fromYMD(year, month, 1).advance(1 + expandN, 'month');
  return ee.ImageCollection(DW_COLLECTION_ID)
    .filterBounds(geom)
    .filterDate(start, end)
    .select(DW_CLASS_NAMES)
    .mean();
}

function labelFromProbs(probImg){
  var label = probImg.toArray().arrayArgmax().arrayGet([0]).rename('label').toInt8();
  var top1  = probImg.reduce(ee.Reducer.max()).rename('top1_prob').unmask(0);
  return label.addBands(top1);
}

function worldCoverDW(geom){
  var WC_TO_DW = {10:1,20:5,30:2,40:4,50:6,60:7,70:8,80:0,90:3,95:1,100:2};
  var wc = ee.ImageCollection(WC_COLLECTION_ID).first().select('Map').clip(geom);
  var wcVals = Object.keys(WC_TO_DW).map(function(k){ return ee.Number.parse(k); });
  var dwVals = Object.keys(WC_TO_DW).map(function(k){ return WC_TO_DW[k]; });
  return wc.remap(wcVals, dwVals).rename('dw_from_wc').toInt8();
}

var YEARS  = ee.List.sequence(startYear, endYear);
var MONTHS = ee.List.sequence(startMonth, endMonth);

var baseProb = ee.ImageCollection.fromImages(
  YEARS.map(function(y){
    return MONTHS.map(function(m){ return monthProb(studyRegion, y, m); });
  }).flatten()
).mean();

var baseLbl = labelFromProbs(baseProb);
var outLbl = baseLbl;
if (EXPAND_MONTHS > 0){
  var expProb = ee.ImageCollection.fromImages(
    YEARS.map(function(y){
      return MONTHS.map(function(m){ return monthProbExpanded(studyRegion, y, m, EXPAND_MONTHS); });
    }).flatten()
  ).mean();
  var expLbl = labelFromProbs(expProb);
  outLbl = outLbl.unmask(expLbl);
}
if (USE_WORLDCOVER){ outLbl = outLbl.unmask(worldCoverDW(studyRegion)); }

var lulc = ee.Image(outLbl.select('label')).clip(studyRegion);
Map.addLayer(lulc, {min:0, max:8, palette: DW_PALETTE}, 'LULC 2019–2024');


// ===================== NBR =====================
var CLOUDY_PCT = 60;
var SHOW_MONTHLY = false;
var DO_EXPORT    = false;

function addNBR(img){
  var nir  = img.select('B8').rename('nir');
  var swir = img.select('B12').resample('bilinear').rename('swir');
  var nbr  = nir.subtract(swir).divide(nir.add(swir)).rename('NBR');
  return img.addBands(nbr);
}

var nbrMean = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyRegion)
  .filterDate(startDate, endDate)
  .filter(seasonalFilter)
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PCT))
  .map(maskS2SR)
  .map(addNBR)
  .select('NBR')
  .mean()
  .clip(studyRegion);

Map.addLayer(nbrMean, {min:-1,max:1,palette:['#7f0000','#b30000','#d7301f','#ef6548','#fdbb84','#fdd49e','#fef0d9']}, 'NBR mean');


// ===================== NDVI (same structure & dates as NBR) =====================
var ndviVis = {min: -0.5, max: 0.9, palette: ['#440154','#31688e','#35b779','#fde725']};

function addNDVI(img){
  var b4 = img.select('B4'); // red
  var b8 = img.select('B8'); // nir
  var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI');
  return img.addBands(ndvi);
}
function meanNDVI(start, end, region){
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate(start, end)
    .filter(seasonalFilter)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PCT))
    .map(maskS2SR)
    .map(addNDVI);
  return col.select('NDVI').mean().clip(region);
}
var ndviMean = meanNDVI(startDate, endDate, studyRegion);
print('NDVI MEAN composite', ndviMean);
Map.addLayer(ndviMean, ndviVis, 'NDVI (mean)');


// ===================== SRTM TERRAIN: elevation, slope, aspect =====================
var dem = ee.Image('USGS/SRTMGL1_003').clip(studyRegion);
var elevation = dem.rename('elevation');
var slope = ee.Terrain.slope(dem).rename('slope');     // degrees
var aspect = ee.Terrain.aspect(dem).rename('aspect');  // degrees (0–360, 0=north)

// Terrain visualization
var elevVis = {min: 0, max: 2500, palette: ['#f7fcf5','#c7e9c0','#74c476','#238b45','#00441b']};
var slopeVis = {min: 0, max: 60, palette: ['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b']};
var aspectPalette = ['#ff0000','#ffff00','#00ff00','#0000ff','#ff0000']; // N→E→S→W→N

//Map.addLayer(elevation, elevVis, 'SRTM Elevation (~30 m)');
//Map.addLayer(slope,     slopeVis, 'SRTM Slope (deg)');
//Map.addLayer(aspect, {min:0,max:360,palette:aspectPalette}, 'SRTM Aspect (deg)');

// (Optional) If you really want 10 m display, uncomment (note: this is interpolation only):
 var dem10    = dem.resample('bilinear').reproject({crs: 'EPSG:4326', scale: 10});
 var slope10  = ee.Terrain.slope(dem10).rename('slope_10m');
 var aspect10 = ee.Terrain.aspect(dem10).rename('aspect_10m');
// Map.addLayer(slope10, slopeVis, 'SRTM Slope (interp 10 m)', false);
// Map.addLayer(aspect10, {min:0,max:360,palette:aspectPalette}, 'SRTM Aspect (interp 10 m)', false);


// ===================== LFMC (unchanged, uses separate mask name) =====================
var years  = ee.List.sequence(startYear, endYear);
var months = ee.List.sequence(startMonth, endMonth);

var worldCover = ee.ImageCollection('ESA/WorldCover/v200').first();
var landMask   = worldCover.select('Map').neq(80).rename('land').clip(studyRegion);

function maskS2SR_LFMC(image) {
  var qa = image.select('QA60');
  var cloudBitMask  = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask)
              .select(['B2','B3','B4','B8','B11'], ['blue','green','red','nir','swir'])
              .copyProperties(image, ['system:time_start']);
}

function getS2Composite(date, geom) {
  var start = ee.Date(date).advance(-15, 'day');
  var end   = ee.Date(date).advance( 15, 'day');
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geom)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .map(maskS2SR_LFMC)
    .median()
    .clip(geom);
}

function getS1Composite(date, geom) {
  var start = ee.Date(date).advance(-30, 'day');
  var end   = ee.Date(date).advance( 30, 'day');
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(geom)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .map(function(i){ return i.select(['VV','VH'], ['vv','vh']).copyProperties(i, ['system:time_start']); })
    .median()
    .clip(geom);
}

function addDerivedBands(img) {
  var ndvi = img.normalizedDifference(['nir','red']).rename('ndvi');
  var ndwi = img.normalizedDifference(['nir','swir']).rename('ndwi');
  var nirv = img.select('nir').multiply(ndvi).rename('nirv');
  var vh_vv = img.select('vh').subtract(img.select('vv')).rename('vh_vv');

  function safeDiv(n, d, name) {
    var eps = ee.Image.constant(1e-6);
    return n.divide(d.abs().max(eps)).rename(name);
  }
  var blue = img.select('blue'), green = img.select('green'), red = img.select('red'),
      nir  = img.select('nir'),  swir  = img.select('swir'),
      vh   = img.select('vh'),   vv    = img.select('vv');

  var vh_blue = safeDiv(vh, blue, 'vh_blue');
  var vh_green= safeDiv(vh, green,'vh_green');
  var vh_red  = safeDiv(vh, red,  'vh_red');
  var vh_nir  = safeDiv(vh, nir,  'vh_nir');
  var vh_swir = safeDiv(vh, swir, 'vh_swir');

  var vv_blue = safeDiv(vv, blue, 'vv_blue');
  var vv_green= safeDiv(vv, green,'vv_green');
  var vv_red  = safeDiv(vv, red,  'vv_red');
  var vv_nir  = safeDiv(vv, nir,  'vv_nir');
  var vv_swir = safeDiv(vv, swir, 'vv_swir');

  return img.addBands([ndvi, ndwi, nirv, vh_vv,
                       vh_blue, vh_green, vh_red, vh_nir, vh_swir,
                       vv_blue, vv_green, vv_red, vv_nir, vv_swir]);
}

function getStatic(geom) {
  var elev  = ee.Image('USGS/SRTMGL1_003').rename('elevation').clip(geom);
  var slope = ee.Terrain.slope(elev).rename('slope').clip(geom);
  return elev.addBands(slope);
}

function renameWithSuffix(img, suffix) {
  var names    = img.bandNames();
  var newNames = names.map(function(n){ return ee.String(n).cat(suffix); });
  return img.select(names, newNames);
}

function allBandsPresent(availableBandNames, neededNames) {
  var flags = ee.List(neededNames).map(function(b) {
    var present = ee.List(availableBandNames).contains(ee.String(b));
    return ee.Number(ee.Algorithms.If(present, 1, 0));
  });
  var sum = ee.Number(ee.List(flags).reduce(ee.Reducer.sum()));
  return sum.eq(neededNames.length);
}

function buildMonthlyLFMC(date, geom) {
  var opt = getS2Composite(date, geom);
  var sar = getS1Composite(date, geom);

  var optBN = opt.bandNames();
  var sarBN = sar.bandNames();

  var hasOpt = allBandsPresent(optBN, ['blue','green','red','nir','swir']);
  var hasSar = allBandsPresent(sarBN, ['vv','vh']);
  var okSensors = hasOpt.and(hasSar);

  return ee.Image(ee.Algorithms.If(
    okSensors,
    (function () {
      var core     = addDerivedBands(opt.addBands(sar)).toFloat();
      var core_t   = renameWithSuffix(core, '_t');
      var static_t = renameWithSuffix(getStatic(geom), '_t');

      var out = ee.Image.constant(100).rename('percent_t')
        .addBands(core_t)
        .addBands(static_t)
        .clip(geom);

      var ndvi_t = out.select('ndvi_t');
      var ndwi_t = out.select('ndwi_t');
      var vh_t   = out.select('vh_t');

      var ndvi_n = ndvi_t.unitScale(0.1, 0.8).clamp(0, 1);
      var ndwi_n = ndwi_t.unitScale(-0.5, 0.5).clamp(0, 1);
      var vh_wet = ee.Image(1).subtract(vh_t.unitScale(-25, -5).clamp(0, 1));

      var lfmc_t = ndvi_n.multiply(0.6)
        .add(ndwi_n.multiply(0.3))
        .add(vh_wet.multiply(0.1))
        .multiply(100)
        .rename('lfmc_t');

      var lfmcMasked = lfmc_t.updateMask(ndvi_t.gte(0.25));

      return lfmcMasked
        .set('ok', 1)
        .set('date', ee.Date(date).format('YYYY-MM-dd'))
        .set('system:time_start', ee.Date(date).millis());
    })(),
    ee.Image([]).set('ok', 0)
                .set('date', ee.Date(date).format('YYYY-MM-dd'))
                .set('system:time_start', ee.Date(date).millis())
  ));
}

var imgList = years.map(function(y){
  y = ee.Number(y);
  var perYearImgs = months.map(function(m){
    m = ee.Number(m);
    var date = ee.Date.fromYMD(y, m, 15);
    return buildMonthlyLFMC(date, studyRegion).set('Y', y).set('M', m);
  });
  return perYearImgs;
}).flatten();

var icRaw = ee.ImageCollection.fromImages(imgList);
var ic = icRaw.filter(ee.Filter.listContains('system:band_names', 'lfmc_t'));

var lfmcMean = ic.select('lfmc_t').mean()
  .updateMask(landMask)
  .clip(studyRegion);

var pctMean = lfmcMean.reduceRegion({
  reducer: ee.Reducer.percentile([10, 40, 70]),
  geometry: studyRegion,
  scale: 10,
  maxPixels: 1e13,
  bestEffort: true,
  tileScale: 8
});
var p10m = ee.Number(ee.Algorithms.If(pctMean.get('lfmc_t_p10'), pctMean.get('lfmc_t_p10'), 70));
var p40m = ee.Number(ee.Algorithms.If(pctMean.get('lfmc_t_p40'), pctMean.get('lfmc_t_p40'), 95));
var p70m = ee.Number(ee.Algorithms.If(pctMean.get('lfmc_t_p70'), pctMean.get('lfmc_t_p70'), 120));

var classesMean = lfmcMean.expression(
  '(lf < t1) ? 0 : (lf < t2) ? 1 : (lf < t3) ? 2 : 3',
  { lf: lfmcMean, t1: p10m, t2: p40m, t3: p70m }
).rename('lfmc_class').clip(studyRegion).toByte();

var classPalette = ['#8b0000', '#ff8c00', '#f0e442', '#1a9850'];
Map.addLayer(classesMean, {min: 0, max: 3, palette: classPalette}, 'LFMC classes (Apr–Nov 2019–2024)');


//-------- LST (Landsat 8/9) --------//
var proj4326 = 'EPSG:4326';
var maxPixels = 1e13;

function maskL1(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1<<3).eq(0)
    .and(qa.bitwiseAnd(1<<4).eq(0))
    .and(qa.bitwiseAnd(1<<2).eq(0))
    .and(qa.bitwiseAnd(1<<5).eq(0));
  return img.updateMask(mask);
}
function toaRefl(img, b){ var m=ee.Number(img.get('REFLECTANCE_MULT_BAND_'+b));
  var a=ee.Number(img.get('REFLECTANCE_ADD_BAND_'+b));
  var se=ee.Number(img.get('SUN_ELEVATION'));
  return img.select('B'+b).multiply(m).add(a).divide(se.multiply(Math.PI/180).sin()).rename('R'+b);
}
function toaRadB10(img){
  var m=ee.Number(img.get('RADIANCE_MULT_BAND_10'));
  var a=ee.Number(img.get('RADIANCE_ADD_BAND_10'));
  return img.select('B10').multiply(m).add(a).rename('Rad10');
}
function btFromRad(img){
  var K1=ee.Number(img.get('K1_CONSTANT_BAND_10'));
  var K2=ee.Number(img.get('K2_CONSTANT_BAND_10'));
  var rad=img.select('Rad10');
  return ee.Image.constant(K2).divide(ee.Image.constant(K1).divide(rad).add(1).log()).rename('BT_K');
}
function emissivityFromNDVI(ndvi){
  var pv = ndvi.subtract(0.2).divide(0.3).clamp(0,1).pow(2);
  return ee.Image(0.986).add(pv.multiply(0.004)).clamp(0.97,0.995).rename('emis');
}
function lstFromBT(btK, emis){
  var lambda_um=10.895, rho=14380.0;
  var corr = ee.Image(1).add(ee.Image(lambda_um).multiply(btK).divide(rho).multiply(emis.log()));
  return btK.divide(corr).subtract(273.15).rename('LST');
}
function processL1(img){
  img=maskL1(img).clip(studyRegion);
  var r4=toaRefl(img,4), r5=toaRefl(img,5);
  var ndvi=r5.subtract(r4).divide(r5.add(r4)).rename('NDVI');
  var rad10=toaRadB10(img);
  var btK=btFromRad(img.addBands(rad10));
  var emis=emissivityFromNDVI(ndvi);
  var lst=lstFromBT(btK, emis);
  return lst.updateMask(lst.gte(-50).and(lst.lte(70)))
            .copyProperties(img,['system:time_start']);
}

var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1')
  .filterBounds(studyRegion).filterDate(startDate,endDate)
  .filter(seasonalFilter).map(processL1);
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1')
  .filterBounds(studyRegion).filterDate(startDate,endDate)
  .filter(seasonalFilter).map(processL1);

var lstMean = L8.merge(L9).mean().rename('LST_mean').toFloat().clip(studyRegion);

print('LST mean stats', lstMean.reduceRegion({
  reducer: ee.Reducer.minMax().combine(ee.Reducer.mean(),'',true),
  geometry: studyRegion, scale:10, maxPixels:maxPixels, bestEffort: true,
  tileScale: 8
}));
Map.addLayer(lstMean,{min:20,max:45,palette:['blue','cyan','yellow','orange','red']},'LST mean °C');


// Final predictor stack
var fuelStack = lulc
  .addBands(nbrMean)
  .addBands(classesMean)
  .addBands(ndviMean)
  .addBands(lstMean)
  //.addBands(elevation)
  .addBands(slope10)
  .addBands(aspect10);

// --- Check projection (resolution) of each predictor layer ---

/*print('LULC:', lulc.projection(), lulc.projection().nominalScale());
print('NBR mean:', nbrMean.projection(), nbrMean.projection().nominalScale());
print('NDVI mean:', ndviMean.projection(), ndviMean.projection().nominalScale());
print('LFMC classes:', classesMean.projection(), classesMean.projection().nominalScale());
print('LST (native):', lstMean.projection(), lstMean.projection().nominalScale());
print('Slope (DEM):', slope10.projection(), slope10.projection().nominalScale());
print('Aspect (DEM):', aspect10.projection(), aspect10.projection().nominalScale());*/




var nonWaterMask = lulc.neq(0);
var filled = fuelStack.updateMask(nonWaterMask).unmask(0);
var predictors = fuelStack.bandNames();
 
var labelStart = ee.Date.fromYMD(startYear, startMonth, 1); // 2015-04-01
var labelEnd   = ee.Date.fromYMD(endYear,   endMonth, 1).advance(1, 'month').advance(-1, 'day'); // 2024-09-30

var firmsIC = ee.ImageCollection('FIRMS')
  .filterBounds(studyRegion)
  .filterDate(labelStart, labelEnd)
  .filter(ee.Filter.calendarRange(startMonth, endMonth, 'month'));

var firmsHigh = firmsIC.map(function(img) {
  return img.select('confidence')
    .gte(60)                         // slightly stricter to reduce noise
    .unmask(0)
    .rename('fire')
    .copyProperties(img, ['system:time_start']);
});


var burnedFirms = firmsHigh
  .max()                          // Any pixel burned during the period
 /// .reproject({ crs: 'EPSG:32610', scale: 30 })
  .clip(studyRegion);
  
var fireBinary  = burnedFirms.gt(0).rename('fire');  

var fireLabelClean = fireBinary
  .focal_mode(1)                                                 // small speckle cleanup
  .updateMask(fireBinary.connectedPixelCount(100, true).gte(2)); 

var fireLabel = fireLabelClean.rename('firelabel');

Map.addLayer(fireLabel, {min:0, max:1, palette:['white','black']}, 'Fire Label (clean, season-aligned)');

var lblHist = fireLabel.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: studyRegion,
  scale: 10,
  maxPixels: 1e13,
  bestEffort: true,
  tileScale: 8
});
print('fireLabel histogram (0=no-fire,1=fire):', lblHist);

var SAMPLE_SCALE = 10;     // keep aligned with your grid
var MAX_PER_CLASS = 1000;  // cap to control memory (tune if needed)
var SPLIT = 0.7;            // 70% train / 30% test

// Masks
var fireMask   = fireLabel.eq(1);
var noFireMask = fireLabel.eq(0);

// Sample across the whole study area for each class
var firePool = filled.mask(fireMask).sample({
  region: studyRegion,
  scale: SAMPLE_SCALE,
  numPixels: MAX_PER_CLASS,
  seed: 2024,
  geometries: true,
  tileScale: 8
}).map(function (f) { return f.set('label', 1); });

var noFirePool = filled.mask(noFireMask).sample({
  region: studyRegion,
  scale: SAMPLE_SCALE,
  numPixels: MAX_PER_CLASS,
  seed: 2025,
  geometries: true,
  tileScale: 8
}).map(function (f) { return f.set('label', 0); });

print('firePool size:', firePool.size());
print('noFirePool size:', noFirePool.size());

// Balance classes (same count fire/no-fire) — deterministic selection
var nKeep = ee.Number(firePool.size()).min(noFirePool.size());
var fireBalanced   = firePool.randomColumn('sel', 777).sort('sel').limit(nKeep);
var noFireBalanced = noFirePool.randomColumn('sel', 888).sort('sel').limit(nKeep);

// Merge and split with one random column
var balancedAll = fireBalanced.merge(noFireBalanced).randomColumn('split', 12345);
var trainingData = balancedAll.filter(ee.Filter.lt('split', SPLIT));
var testData     = balancedAll.filter(ee.Filter.gte('split', SPLIT));

// Inspect counts (optional)
print('Training samples (balanced total):', trainingData.size());
print('Test samples (balanced total):',      testData.size());
print('Train class histogram (0=no-fire,1=fire):', trainingData.aggregate_histogram('label'));
print('Test class histogram  (0=no-fire,1=fire):', testData.aggregate_histogram('label'));

// === Train & evaluate (rest stays the same) ===
var nFeat = predictors.size();
var mtry  = ee.Number(nFeat).sqrt().floor();

var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 400,
  variablesPerSplit: mtry,
  minLeafPopulation: 6,
  bagFraction: 0.7,
  seed: 99
}).train({
  features: trainingData,
  classProperty: 'label',
  inputProperties: predictors
});

print('RF explain:', classifier.explain());

// evaluate at default 0.5
var testResult = testData.classify(classifier);
var confusionMatrix = testResult.errorMatrix('label', 'classification');
print('Confusion Matrix:', confusionMatrix);
print('Accuracy:', confusionMatrix.accuracy());
print('Kappa:', confusionMatrix.kappa());
print('Recall:', confusionMatrix.producersAccuracy());
print('Precision:', confusionMatrix.consumersAccuracy());

// Feature Importance
var rawImportance = ee.Dictionary(classifier.explain().get('importance'));

// Get feature names
var keys = rawImportance.keys();

// Get importance values for each key
var values = keys.map(function(k) {
  return rawImportance.get(k);
});

// Compute total sum and extract as a scalar number
var total = ee.Number(ee.Array(values).reduce(ee.Reducer.sum(), [0]).get([0]));

// Normalize values: divide each by the total
var normalizedImportance = ee.Dictionary.fromLists(
  keys,
  values.map(function(v) {
    return ee.Number(v).divide(total);
  })
);
// Rename dictionary as ee.Dictionary
var renameDict = ee.Dictionary({
  'lulc': 'LULC',
  'nbrMean': 'NBR',
  'classesMean': 'Fuel Moisture',
  'ndviMean': 'NDVI',
  'lstMean': 'LST',
  'slope': 'Slope',
  'aspect': 'Aspect',
  'elevation': 'Elevation'
});

// Map feature names and create FeatureCollection
var importanceList = keys.map(function(k) {
  k = ee.String(k);
  var label = ee.Algorithms.If(renameDict.contains(k), renameDict.get(k), k);
  return ee.Feature(null, {
    'feature': label,
    'importance': normalizedImportance.get(k)
  });
});

var importanceFC = ee.FeatureCollection(importanceList);

// Print FeatureCollection (note lowercase variable name)
print('Normalized Feature Importance:', importanceFC);

// Chart
var chart = ui.Chart.feature.byFeature(importanceFC, 'feature', ['importance'])
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Normalized Feature Importance (Random Forest)',
    hAxis: { title: 'Feature' },
    vAxis: { title: 'Importance (Fraction)', minValue: 0 },
    legend: { position: 'none' },
    bar: { groupWidth: '90%' },
    colors: ['#1f77b4']
  });

print(chart);



// ---------------- Map (probability smoothed for display only) ---------
var probability = filled.classify(classifier.setOutputMode('PROBABILITY'), 'probability');
var susceptibility = probability
  .convolve(ee.Kernel.gaussian({radius: 4, sigma: 2, units: 'pixels'}))
  .rename('susceptibility')
  .clip(studyRegion);

Map.addLayer(susceptibility, {min:0, max:1, palette:['green','lightgreen','yellow','orange','red']}, 'Susceptibility');

var riskClasses = susceptibility.expression(
  "prob <= 0.2 ? 1 : (prob <= 0.4 ? 2 : (prob <= 0.6 ? 3 : (prob <= 0.8 ? 4 : 5)))",
  {'prob': susceptibility.select('susceptibility')}
).rename('risk_class').clip(studyRegion);

Map.addLayer(riskClasses, {min:1, max:5, palette:['green','lightgreen','yellow','orange','red']}, 'Risk Classes');

// ---- Legends for Susceptibility and Risk ----
var legendPanel = ui.Panel({style: {position: 'bottom-right', padding: '8px'}});

// Susceptibility legend
legendPanel.add(ui.Label('Susceptibility', {fontWeight: 'bold'}));
var suscColors = ['green','lightgreen','yellow','orange','red'];
var suscLabels = ['0.0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1.0'];
suscColors.forEach(function(c, i){
  legendPanel.add(ui.Panel([
    ui.Label('', {backgroundColor: c, padding: '8px', margin: '2px', border: '1px solid black'}),
    ui.Label(suscLabels[i], {margin: '4px 0 4px 6px'})
  ], ui.Panel.Layout.Flow('horizontal')));
});

legendPanel.add(ui.Label('')); // spacer

// Risk legend (classes 1–5)
/*legendPanel.add(ui.Label('Risk Classes', {fontWeight: 'bold'}));
var riskColors = ['green','lightgreen','yellow','orange','red'];
var riskNames  = ['Very Low','Low','Moderate','High','Very High'];
riskColors.forEach(function(c, i){
  legendPanel.add(ui.Panel([
    ui.Label('', {backgroundColor: c, padding: '8px', margin: '2px', border: '1px solid black'}),
    ui.Label('Class ' + (i+1) + ' — ' + riskNames[i], {margin: '4px 0 4px 6px'})
  ], ui.Panel.Layout.Flow('horizontal')));
});*/

ui.root.add(legendPanel);

// ---- Export risk classes (1–5) to Google Drive at 30 m resolution ----
/*var classExport = riskClasses
  .unmask(0)             // fill masked pixels with 0 (background)
  .resample('nearest')   // preserve class labels when scaling
  .toByte()              // store as 8-bit integer (0..255)
  .rename('risk_class');

Export.image.toDrive({
  image: classExport,
  description: 'Risk_Classes_30m',
  folder: 'GEE_Exports',
  fileNamePrefix: 'risk_classes_30m_' + countryName + '_' + year,
  region: studyRegion,
  scale: 30,                                 // 30 m
  maxPixels: 1e13,
  crs: 'EPSG:32636',                         // UTM Zone 36N (meters) for Cyprus
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true,                    // COG for better GIS streaming
    noData: 0                                // QGIS/ArcGIS will treat 0 as NoData
  }
});*/

// Optional: also export the continuous susceptibility probability at 30 m
Export.image.toDrive({
  image: susceptibility, 
  description: 'G22_Susceptibility_30m',
  folder: 'GEE_Exports',
  fileNamePrefix: 'G22_susceptibility30m_',
  region: studyRegion,
  scale: 30,
  maxPixels: 1e13,
  crs: 'EPSG:32636'
});

