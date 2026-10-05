var countryName = 'Limassol';
var year = 2025;

// Center map on Cyprus
// 1. Center map and display boundary
Map.centerObject(studyRegion, 12);
Map.addLayer(studyRegion, {color: 'blue'}, 'Study Region');




var startYear = 2019;
var endYear = 2024;
var startMonth = 4; // April
var endMonth = 9; // Sep
var startDate = ee.Date.fromYMD(startYear, startMonth, 1);
var endDate = ee.Date.fromYMD(endYear, endMonth, 1).advance(1, 'month').advance(-1, 'day'); // last day of endMonth
var seasonalFilter = ee.Filter.calendarRange(startMonth, endMonth, 'month');


// Load MODIS MCD12Q1 for 2023
var modisLC = ee.ImageCollection("MODIS/061/MCD12Q1")
  .filterDate(startDate, endDate)
  .select('LC_Type1')
  .map(function(img) {
    return img.clip(studyRegion).set('system:time_start', img.date().millis());
  });
var modeLC = modisLC.mode();


// Reclassify MODIS IGBP classes to fuel types
var fuelRemap = modeLC.remap(
  [17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  // MODIS Classes
  [5, 4, 3, 4, 3, 3, 2, 2, 2, 2, 1, 5, 0, 0, 0, 5, 0]        // Corresponding Fuel Classes
).rename('fuel_class');

// Load Landsat 8 Surface Reflectance from 2018-2024 and apply scale factors
var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(studyRegion)
  .filterDate(startDate, endDate)
  .filter(seasonalFilter) 
  .filter(ee.Filter.lt('CLOUD_COVER', 10))
  .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
  .median()
  .multiply(0.0000275).add(-0.2); // Scale to reflectance
 // .clip(studyRegion);  

// Calculate NDVI
var ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');

// Load SRTM DEM data (for slope and aspect)
var dem = ee.Image('USGS/SRTMGL1_003').clip(studyRegion);
var slope = ee.Terrain.slope(dem).rename('slope');
var aspect = ee.Terrain.aspect(dem).rename('aspect');

// TRI
var kernel = ee.Kernel.square({radius: 1, units: 'pixels', normalize: false});
var meanElev = dem.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: kernel});
var tri = dem.subtract(meanElev).abs().rename('tri');

// Reprojection 
var reprojectTo250m = function(image) {
  return image.reproject({
      crs: 'EPSG:4326', //32610
      scale: 30
    });
};

var landsat250 = reprojectTo250m(landsat);
var ndvi250 = reprojectTo250m(ndvi);
var slope250 = reprojectTo250m(slope);
var aspect250 = reprojectTo250m(aspect);
var tri250 = reprojectTo250m(tri);

// Aspect & TRI Visualization
var aspectPalette = [
  'red',       // North (0°)
  'yellow',    // East (90°)
  'green',     // South (180°)
  'blue',      // West (270°)
  'red'        // Back to North (360°)
];

Map.addLayer(aspect250, {min: 0, max: 360, palette: aspectPalette}, 'Aspect');

Map.addLayer(tri250, {
  min: 0,
  max: 10.8,
  palette: ['#f7fcb9', '#addd8e', '#31a354', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026', '#800026', '#f7f7f7']
}, 'TRI (0–10.8)');

// Stacking Terrain Features
var terrainStack = slope250.addBands(aspect250).addBands(tri250);
var terrain_Stack = terrainStack .toFloat();  // Converts all bands to Float32*/

// --- NDVI texture for structure separation (broadleaf vs shrubs) ---
var ndvi8bit = ndvi250.multiply(255).toByte(); // GLCM expects 8-bit
var ndviTex  = ndvi8bit.glcmTexture({size: 3}).select('NDVI_contrast');
// Normalize texture to 0–1 for stable RF input
var NDVI_contrast = ndviTex.unitScale(0, 50).clamp(0, 1).rename('NDVI_contrast');


//  Stacking all features (Add slope and aspect to Landsat data)
var landsatStack = landsat250
  .addBands(ndvi250)
  .addBands(NDVI_contrast)
  .addBands(slope250)
  .addBands(aspect250)
  .addBands(tri250);
var landsatStackNorm = landsatStack.unitScale(0, 1); // basic normalization between 0-1

var fuelRemap250 = reprojectTo250m(fuelRemap);
var fuelRemapFilled = fuelRemap250.unmask(0); // Fill missing areas with "0" (non-fuel or unknown class)


// Convert the fuel class image to a FeatureCollection of points
var fuelPoints = fuelRemapFilled.addBands(landsatStackNorm).stratifiedSample({
    numPoints: 1800,
    classBand: 'fuel_class',
    classValues: [0, 1, 2, 3, 4, 5], 
    classPoints: [120, 280, 280, 340, 220, 120],
    region: studyRegion,
    scale: 250,
    seed: 123,
    geometries: true,
    tileScale: 8
});

//print('Sample Points:', fuelPoints.size());

// Split training data into training and testing sets
var withRandom = fuelPoints.randomColumn('random', 50);
var trainSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var testSet = withRandom.filter(ee.Filter.gte('random', 0.7));

// Define input bands for classification (Landsat + NDVI + slope + aspect)
var inputBands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI', 'NDVI_contrast', 'slope', 'aspect','tri'];

// RF Classifier (deterministic)
var rf_classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 160,
    seed: 42
  }).train({
    features: trainSet,
    classProperty: 'fuel_class',
    inputProperties: inputBands
  });

// Apply RF classifier to Landsat data
var rf_classified = landsatStack.classify(rf_classifier);

// Smoothing Filter
var modeFilter = ee.Kernel.square({radius: 1});
var smoothed_rf = rf_classified.focal_mode({kernel: modeFilter, iterations: 1});

var smoothed_rf250 = reprojectTo250m(smoothed_rf);
Map.addLayer(smoothed_rf250, {min: 0, max: 5, palette: ['gray', 'yellow', 'brown', 'lightgreen', 'darkgreen', 'blue']}, 'Smoothed RF Classification');

// Accuracy Assessment
var rfValidated = testSet.classify(rf_classifier);
var rfMatrix = rfValidated.errorMatrix('fuel_class', 'classification');
print('RF Confusion Matrix:', rfMatrix);
print('RF Overall Accuracy:', rfMatrix.accuracy());
print('RF Kappa:', rfMatrix.kappa());
print('Recall:', rfMatrix.producersAccuracy());
print('Precision:', rfMatrix.consumersAccuracy());


// Visualize the classified results
//Map.addLayer(rf_classified, {min: 0, max: 5, palette: ['gray', 'yellow', 'brown', 'lightgreen', 'darkgreen', 'blue']}, 'RF Classification');


var masked_rf = smoothed_rf250.clip(studyRegion).selfMask();

//-------------------------------------------------------------//
// Fuel Load Estimation
// Load GEDI L4A image
var gedi = ee.Image("LARSE/GEDI/GEDI04_B_002") // Image- Already shows a product based on 2019-2020 years
  .select(['MU', 'QF'])
  .clip(studyRegion);
  
// Select MU and QF bands
var agb = gedi.select('MU');  // Mean Biomass in Mg/ha
var quality = gedi.select('QF'); // Quality Flag 

// Apply quality mask (QF == 2) to filter out low quality data
var agbMasked = agb.updateMask(quality.eq(2));

// Add AGB band to existing features
var agbBand = agbMasked.rename('AGB');
var featuresWithAGB = landsatStackNorm.addBands(agbBand);

/*// Sample training data for regression to predict values where AGB is missing
var trainingSamples = featuresWithAGB.select(['SR_B5','SR_B4','SR_B6','SR_B7','NDVI','slope','aspect','tri','AGB'])
  .sample({
    region: studyRegion,
    scale: 250,
    numPixels: 4000,
    seed: 35
  });*/
  
var agbMask = agbBand.mask(); // mask of valid GEDI
var featuresOnAGB = landsatStackNorm
  .addBands(agbBand)           // has AGB
  .updateMask(agbMask);        // keep only valid AGB pixels

// ---- UPDATED: include SWIR & TRI (correlate with biomass/structure) ----
var agbPredictors = ['SR_B4','SR_B5','SR_B6','SR_B7','NDVI','slope','aspect','tri','AGB'];

// Larger sample for stability; filter not-null explicitly
var trainingSamples = featuresOnAGB.select(agbPredictors)
  .sample({
    region: studyRegion,
    scale: 250,
    numPixels: 8000,
    seed: 35
  })
  .filter(ee.Filter.notNull(agbPredictors));
  
//print('Training sample size:', trainingSamples.size());

// Split the data: 70% training, 30% validation
var split = 0.7;
var withRandom = trainingSamples.randomColumn('random', 123);
var training = withRandom.filter(ee.Filter.lt('random', split));
var validation = withRandom.filter(ee.Filter.gte('random', split));

var agbRF = ee.Classifier.smileRandomForest({
    numberOfTrees: 200,
    seed: 42
  })
  .setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'AGB',
    inputProperties: ['SR_B5','SR_B4','SR_B6','SR_B7','NDVI','slope','aspect','tri']
  });
  


// Predict AGB everywhere features exist (your original way)
var agbPredicted = landsatStackNorm
  .select(['SR_B4','SR_B5','SR_B6','SR_B7','NDVI','slope','aspect','tri'])
  .classify(agbRF)
  .rename('AGB_RF');

// Fill missing GEDI with RF predictions (unchanged)
var agbFilledRF = agbMasked.unmask(agbPredicted);
var agbFinal250m = reprojectTo250m(agbFilledRF);

// -------- Validation metrics (NULL-SAFE) --------
var validationPred = validation.classify(agbRF);

// Keep rows where both truth & pred exist
var valClean = validationPred.filter(
  ee.Filter.and(
    ee.Filter.notNull(['AGB']),
    ee.Filter.notNull(['classification'])
  )
);

// Map errors safely
var validatedWithErrors = valClean.map(function(f) {
  var predicted = ee.Number(f.get('classification'));
  var actual    = ee.Number(f.get('AGB'));
  var error     = predicted.subtract(actual);
  return f.set({
    error: error,
    absError: error.abs(),
    sqError: error.pow(2)
  });
});

// Metrics
var mae = validatedWithErrors.reduceColumns(
  ee.Reducer.mean(), ['absError']
).get('mean');

var mse = validatedWithErrors.reduceColumns(
  ee.Reducer.mean(), ['sqError']
).get('mean');

var agbVar = valClean.reduceColumns(
  ee.Reducer.variance(), ['AGB']
).get('variance');

var rmse = ee.Number(mse).sqrt();
var r2   = ee.Number(1).subtract(ee.Number(mse).divide(agbVar));

// Print
print('MAE (Mg/ha):', mae);
print('RMSE (Mg/ha):', rmse);
print('R²:', r2);

// Map layer (unchanged)
Map.addLayer(agbFinal250m, {min:0, max:90, palette:['white','yellowgreen','green','darkgreen']}, 'Filled AGB Layer (250m)');

Export.image.toDrive({
  image: agbFinal250m,
  description: 'AGB_estimation_sacramento',
  folder: 'GEE_Exports',  // Optional: specify folder
  fileNamePrefix: 'AGB_estimation_sacramento',
  region: studyRegion,
  scale: 30,
  crs: 'EPSG:4326',  // Optional: specify CRS if needed
  fileFormat: 'GeoTIFF'
});
// --------- Fuel Moisture ----------
var years  = ee.List.sequence(startYear, endYear);
var months = ee.List.sequence(startMonth, endMonth);

// --------------------------- LAND MASK -----------------------------
var worldCover = ee.ImageCollection('ESA/WorldCover/v200').first();
var landMask   = worldCover.select('Map').neq(80).rename('land').clip(studyRegion);

// =========================== HELPERS ==============================
function maskS2SR(image) {
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
    .map(maskS2SR)
    .median()
    .clip(geom); // no landMask here
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

// Check that all needed bands exist in available band list (no reduceRegion used)
function allBandsPresent(availableBandNames, neededNames) {
  var flags = ee.List(neededNames).map(function(b) {
    var present = ee.List(availableBandNames).contains(ee.String(b));
    return ee.Number(ee.Algorithms.If(present, 1, 0));
  });
  var sum = ee.Number(ee.List(flags).reduce(ee.Reducer.sum()));
  return sum.eq(neededNames.length); // ee.Boolean
}

// ================== MONTHLY LFMC BUILDER (NO REDUCE REGIONS) ==================
function buildMonthlyLFMC(date, geom) {
  var opt = getS2Composite(date, geom);
  var sar = getS1Composite(date, geom);

  var optBN = opt.bandNames();
  var sarBN = sar.bandNames();

  var hasOpt = allBandsPresent(optBN, ['blue','green','red','nir','swir']);
  var hasSar = allBandsPresent(sarBN, ['vv','vh']);
  var okSensors = hasOpt.and(hasSar);

  // If both sensors present, build LFMC_t masked to vegetation; else return empty
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

      // LFMC proxy (uncalibrated)
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

      // Vegetation mask only; NO landMask, NO reduceRegion
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

// ================== BUILD COLLECTION (2015–2024 Apr–Sep) ==================
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
// Keep only images that actually have lfmc_t band
var ic = icRaw.filter(ee.Filter.listContains('system:band_names', 'lfmc_t'));

print('Valid monthly images kept:', ic.size());

// ===================== AGGREGATION & DISPLAY =======================
// Mean LFMC across valid months; now it's safe to apply landMask
var lfmcMean = ic.select('lfmc_t').mean()
  .updateMask(landMask)
  .clip(studyRegion);

// Percentile thresholds & classes (computed ONCE)
var pctMean = lfmcMean.reduceRegion({
  reducer: ee.Reducer.percentile([10, 40, 70]),
  geometry: studyRegion,
  scale: 250,
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
).rename('lfmc_class').toByte().clip(studyRegion);

// “Dry union”: threshold each monthly LFMC by the single global p10 (p10m) and union
var dryIC = ic.select('lfmc_t').map(function(img){
  return img.lt(p10m).toByte().rename('dry').copyProperties(img, ['system:time_start', 'date', 'Y', 'M']);
});
var unionDry = ee.ImageCollection(dryIC).max()
  .updateMask(landMask)
  .clip(studyRegion);

// Map layers
var classPalette = ['#8b0000', '#ff8c00', '#f0e442', '#1a9850'];
Map.addLayer(classesMean, {min: 0, max: 3, palette: classPalette}, 'LFMC classes (Apr–Sep 2015–2024)');
Map.addLayer(unionDry.selfMask(), {palette: ['#ff0000']}, 'Dry areas union (Apr–Sep 2015–2024)');


//------------- Susceptibility -----------------//
var baseProj = agbFinal250m.projection();
var classesMean_250m = classesMean.resample('bilinear').reproject(baseProj).clip(studyRegion);

var FuelType = smoothed_rf250.rename('FuelType').toFloat().reproject(baseProj);
var AGB      = agbFinal250m.rename('AGB').toFloat().reproject(baseProj);
var FM       = classesMean.rename('FM').toFloat().reproject(baseProj);

// ⬇ add both proxies to the stack, plus new predictors (NDVI mean, LST mean, precipitation seasonal sum)
// --- NDVI seasonal mean from Sentinel-2 (Apr–Sep across years) ---
var s2NDVImean = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(studyRegion)
  .filterDate(startDate, endDate)
  .filter(seasonalFilter)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
  .map(maskS2SR)
  .map(function(img){
    return img.normalizedDifference(['nir','red']).rename('NDVI');
  })
  .mean()
  .rename('NDVI_mean')
  .toFloat()
  .reproject(baseProj)
  .clip(studyRegion);

// --- MODIS LST (8-day, daytime) mean in °C (Apr–Sep across years) ---
function maskMOD11A2(img){
  var lst = img.select('LST_Day_1km').multiply(0.02).subtract(273.15); // Kelvin to Celsius
  var qa  = img.select('QC_Day');
  var good = qa.bitwiseAnd(1).eq(0); // basic good-quality filter
  return lst.updateMask(good);
}
var lstMean = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(startDate, endDate)
  .filter(ee.Filter.calendarRange(startMonth, endMonth, 'month'))
  .map(maskMOD11A2)
  .mean()
  .rename('LST_mean')
  .toFloat()
  .reproject(baseProj)
  .clip(studyRegion);

// --- CHIRPS precipitation: seasonal total per year → mean across years (mar–Sep) ---
var yearList = ee.List.sequence(startYear, endYear);
var perYearSeason = ee.ImageCollection(yearList.map(function(y){
  y = ee.Number(y);
  var ys = ee.Date.fromYMD(y, startMonth, 1);
  var ye = ee.Date.fromYMD(y, endMonth, 1).advance(1,'month').advance(-1,'day');
  var sum = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(studyRegion)
    .filterDate(ys, ye)
    .select('precipitation')
    .sum()
    .rename('Prcp_season');
  return sum.set('year', y);
}));
var prcpSeasonMean = perYearSeason.mean()
  .rename('Prcp_cumsum')
  .toFloat()
  .reproject(baseProj)
  .clip(studyRegion);

// Final predictor stack
var fuelStack = FuelType
  .addBands(AGB)
  .addBands(FM)
  .addBands(s2NDVImean)
  .addBands(lstMean)
 // .addBands(prcpSeasonMean)
  .addBands(slope250)
  .addBands(aspect250);

var nonWaterMask = FuelType.neq(5);
var filled = fuelStack.updateMask(nonWaterMask).unmask(0);
var predictors = fuelStack.bandNames();
 
// Fire label Generation
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
  .updateMask(fireBinary.connectedPixelCount(100, true).gte(2)); // <-- FIX HERE

var fireLabel = reprojectTo250m(fireLabelClean).rename('firelabel');

Map.addLayer(fireLabel, {min:0, max:1, palette:['white','black']}, 'Fire Label (clean, season-aligned)');

var lblHist = fireLabel.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: studyRegion,
  scale: 250,
  maxPixels: 1e9
});
print('fireLabel histogram (0=no-fire,1=fire):', lblHist);

// Export fire label to Google Drive
Export.image.toDrive({
  image: fireLabel,
  description: 'fire_label_Sacramento_2015_2024',
  folder: 'GEE_exports',       // optional folder in Drive
  fileNamePrefix: 'fire_label_Sacramento_2015_2024',
  region: studyRegion,
  scale: 30,                   // your Sentinel-2 grid (or 250 if you want)
  crs: 'EPSG:4326',
  maxPixels: 1e13
});


var SAMPLE_SCALE = 250;     // keep aligned with your grid
var MAX_PER_CLASS = 12000;  // cap to control memory (tune if needed)
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

// === Training===
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

// Testing & Evaluation
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
  'AGB': 'AGB',
  'FuelType': 'Fuel Type',
  'FM': 'Fuel Moisture'
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
  description: 'Susceptibility_Prob_250m',
  folder: 'GEE_Exports',
  fileNamePrefix: 'susceptibility_prob_250m_' + countryName + '_' + year,
  region: studyRegion,
  scale: 250,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});

Map.centerObject(table, 12);
Map.addLayer(table, {color: 'blue'}, 'table');
