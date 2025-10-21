import React, { useMemo, useState } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';

// Emax model function with H=1 (normal scale parameters)
function emaxModel(dose, E0, Emax, ED50) {
  const h = 1; // Fixed Hill coefficient
  const num = Emax * Math.pow(dose, h);
  const den = Math.pow(ED50, h) + Math.pow(dose, h);
  return E0 - (den > 0 ? num / den : 0);
}

// Emax model function with log-transformed parameters
// Takes logEmax and logED50 as inputs for numerical stability
function emaxModelLog(dose, E0, logEmax, logED50) {
  const Emax = Math.exp(logEmax);
  const ED50 = Math.exp(logED50);
  return emaxModel(dose, E0, Emax, ED50);
}

// Add random noise to simulated data
// Generate standard normal random variable using Box-Muller transform
function randn() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Add normally distributed noise to a value
function addNoise(value, noiseLevel = 5) {
  return value + randn() * noiseLevel;
}

// Calculate expected Fisher information matrix for the current parameters
// This gives us the expected precision of parameter estimates at the current design
function calculateExpectedInformation(doses, params, sigma = 5) {
  const E0 = 100;
  const { Emax, ED50 } = params;
  const logEmax = Math.log(Emax);
  const logED50 = Math.log(ED50);
  
  // Build Fisher information matrix on log scale
  // I_ij = (1/sigma^2) * sum_k [ (dy_k/dtheta_i) * (dy_k/dtheta_j) ]
  let I11 = 0; // d/dlogEmax
  let I12 = 0; // cross term
  let I22 = 0; // d/dlogED50
  
  for (const dose of doses) {
    const x = dose;
    // Analytical derivatives w.r.t. log parameters
    // dy/d(logEmax) = -(x/(ED50+x)) * Emax
    const dy_dlogEmax = -(x / (ED50 + x)) * Emax;
    
    // dy/d(logED50) = (Emax * x * ED50) / (ED50 + x)^2
    const dy_dlogED50 = (Emax * x * ED50) / Math.pow(ED50 + x, 2);
    
    I11 += dy_dlogEmax * dy_dlogEmax;
    I12 += dy_dlogEmax * dy_dlogED50;
    I22 += dy_dlogED50 * dy_dlogED50;
  }
  
  // Scale by 1/sigma^2
  const sigma2 = sigma * sigma;
  I11 /= sigma2;
  I12 /= sigma2;
  I22 /= sigma2;
  
  // Invert to get covariance matrix on log scale
  const det = I11 * I22 - I12 * I12;
  
  if (det <= 0 || I11 <= 0 || I22 <= 0) {
    console.warn('Fisher information matrix not positive definite');
    return null;
  }
  
  // Covariance on log scale
  const var_logEmax = I22 / det;
  const var_logED50 = I11 / det;
  const cov_logEmax_logED50 = -I12 / det;
  
  // Transform to normal scale using delta method
  const var_Emax = Emax * Emax * var_logEmax;
  const var_ED50 = ED50 * ED50 * var_logED50;
  const cov_Emax_ED50 = Emax * ED50 * cov_logEmax_logED50;
  
  return {
    var_Emax,
    var_ED50,
    cov_Emax_ED50
  };
}

// Generate samples from bivariate normal distribution using Cholesky decomposition
// This properly accounts for correlation between Emax and ED50
function sampleMultivariateNormal(mean, covMatrix, numSamples = 100) {
  const { var_Emax, var_ED50, cov_Emax_ED50 } = covMatrix;
  
  // Cholesky decomposition of covariance matrix
  // For 2x2: L = [[L11, 0], [L21, L22]]
  const L11 = Math.sqrt(var_Emax);
  const L21 = cov_Emax_ED50 / L11;
  const L22 = Math.sqrt(var_ED50 - L21 * L21);
  
  if (isNaN(L22) || L22 <= 0) {
    console.warn('Covariance matrix not positive definite');
    return [];
  }
  
  const samples = [];
  for (let i = 0; i < numSamples; i++) {
    // Generate two independent standard normal random variables (Box-Muller transform)
    const u1 = Math.random();
    const u2 = Math.random();
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
    
    // Transform to correlated samples using Cholesky
    const Emax_sample = mean.Emax + L11 * z1;
    const ED50_sample = mean.ED50 + L21 * z1 + L22 * z2;
    
    samples.push({ Emax: Emax_sample, ED50: ED50_sample });
  }
  
  return samples;
}

function fmt(x, digits = 2) {
  return Number(x).toFixed(digits);
}

function SliderRow({ label, min, max, step, value, onChange }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-sm tabular-nums font-semibold">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-indigo-600"
      />
      <div className="flex justify-between text-xs text-gray-500">
        <span>{fmt(min)}</span>
        <span>{fmt(max)}</span>
      </div>
    </div>
  );
}

export default function App() {
  // User-adjusted parameters (E0 fixed at 100)
  const [params, setParams] = useState({ Emax: 100, ED50: 0.5 });
  
  const E0 = 100; // Fixed baseline
  
  // Dose input
  const [doseInput, setDoseInput] = useState('');
  
  // Show/hide drug characteristics
  const [showParams, setShowParams] = useState(false);

  const updateParam = (key, value) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Parse doses from input
  const doses = useMemo(() => {
    try {
      return doseInput
        .split(',')
        .map(d => parseFloat(d.trim()))
        .filter(d => !isNaN(d) && d >= 0);
    } catch {
      return [];
    }
  }, [doseInput]);

  // Generate smooth curve for current parameters (dense for smooth hovering)
  const curve = useMemo(() => {
    const maxDose = Math.max(...doses, 5);
    const points = Array.from({ length: 201 }, (_, i) => (i / 200) * maxDose);
    return points.map(x => ({
      x,
      y: Math.max(0, Math.min(120, emaxModel(x, E0, params.Emax, params.ED50)))
    }));
  }, [params, doses]);

  // Calculate expected information matrix for current parameters and doses
  const expectedInfo = useMemo(() => {
    if (doses.length === 0) return null;
    return calculateExpectedInformation(doses, params, 7);
  }, [doses, params]);

  // Calculate %RSE and determine if precision is poor
  const precisionMetrics = useMemo(() => {
    if (!expectedInfo) return null;
    
    const rseEmax = (Math.sqrt(expectedInfo.var_Emax) / params.Emax) * 100;
    const rseED50 = (Math.sqrt(expectedInfo.var_ED50) / params.ED50) * 100;
    const isPoor = rseEmax > 50 || rseED50 > 50;
    
    return {
      rseEmax,
      rseED50,
      isPoor,
      color: isPoor ? '#ef4444' : '#8884d8' // red if poor, blue if good
    };
  }, [expectedInfo, params]);

  // Generate confidence bands based on expected information
  const confidenceBands = useMemo(() => {
    if (!expectedInfo) return null;
    
    const maxDose = Math.max(...doses, 5);
    const points = Array.from({ length: 201 }, (_, i) => (i / 200) * maxDose);
    
    // Sample from multivariate normal distribution
    const samples = sampleMultivariateNormal(params, expectedInfo, 200);
    
    if (samples.length === 0) return null;
    
    return points.map(x => {
      const yFit = emaxModel(x, E0, params.Emax, params.ED50);
      
      // Calculate predictions for all parameter samples at this dose
      const predictions = samples.map(sample => 
        emaxModel(x, E0, sample.Emax, sample.ED50)
      );
      
      // Calculate percentiles (use 5th and 95th for 90% CI)
      predictions.sort((a, b) => a - b);
      const lowerIdx = Math.floor(predictions.length * 0.05);
      const upperIdx = Math.floor(predictions.length * 0.95);
      
      return {
        x,
        y: yFit,
        upper: Math.max(0, Math.min(120, predictions[upperIdx])),
        lower: Math.max(0, Math.min(120, predictions[lowerIdx]))
      };
    });
  }, [expectedInfo, params, doses]);

  return (
    <div className="min-h-screen w-full bg-white text-gray-900">
      <div className="max-w-6xl mx-auto p-6">
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">The Importance of Good Data</h1>
          <p className="text-gray-600 mt-2">
            Explore how experimental design affects prediction precision
            </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Left panel - Controls */}
          <div className="md:col-span-1 space-y-4">
            {/* Dose Input */}
            <div className="bg-gray-50 rounded-2xl p-4 shadow-sm border border-gray-200">
              <h3 className="font-semibold mb-2">Dose Design</h3>
              <input
                type="text"
                value={doseInput}
                onChange={(e) => setDoseInput(e.target.value)}
                placeholder="e.g., 0.5, 1, 2, 3, 4"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              <p className="text-xs text-gray-500 mt-1">Enter comma-separated dose values (mg/kg)</p>
            </div>

            {/* Parameter Sliders - Collapsible */}
            <div className="bg-gray-50 rounded-2xl p-4 shadow-sm border border-gray-200">
              <button
                onClick={() => setShowParams(!showParams)}
                className="w-full flex items-center justify-between font-semibold mb-2"
              >
                <span>Drug Characteristics</span>
                <span className="text-lg">{showParams ? '▼' : '▶'}</span>
              </button>
              
              {showParams && (
                <>
                  <div className="mb-3 text-sm bg-white border border-gray-200 rounded-xl p-3">
                    <div className="font-medium mb-1">Equation (H = 1, E0 = 100)</div>
                    <div className="font-mono text-xs">
                      y = 100 − (Emax·x)/(ED50 + x)
                    </div>
                  </div>

                  <div className="space-y-4">
                    <SliderRow
                      label="Maximal Effect"
                      min={0}
                      max={100}
                      step={0.5}
                      value={params.Emax}
                      onChange={(v) => updateParam('Emax', v)}
                    />
                    <SliderRow
                      label="Potency (ED50)"
                      min={0.1}
                      max={10}
                      step={0.1}
                      value={params.ED50}
                      onChange={(v) => updateParam('ED50', v)}
                    />
                  </div>
                </>
              )}
            </div>

            {/* Expected Precision */}
            {expectedInfo && precisionMetrics && (
              <div className="bg-gray-50 rounded-2xl p-4 shadow-sm border border-gray-200">
                <h3 className="font-semibold mb-2">Expected Precision</h3>
                <p className="text-xs text-gray-500 mb-2">
                  At current design:
                </p>
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span>%RSE(Emax):</span>
                    <span className={`font-mono ${precisionMetrics.rseEmax > 50 ? 'text-red-600 font-semibold' : ''}`}>
                      {fmt(precisionMetrics.rseEmax)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>%RSE(ED50):</span>
                    <span className={`font-mono ${precisionMetrics.rseED50 > 50 ? 'text-red-600 font-semibold' : ''}`}>
                      {fmt(precisionMetrics.rseED50)}%
                    </span>
                  </div>
                </div>
                {precisionMetrics.isPoor && (
                  <p className="text-xs text-red-600 mt-2 font-semibold">
                    ⚠ Poor precision: %RSE &gt; 50%
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Right panel - Chart */}
          <div className="md:col-span-2 bg-white rounded-2xl p-4 shadow-sm border border-gray-200">
            <div className="h-[500px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={curve} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={[0, 'auto']}
                    label={{ value: 'Dose (mg/kg)', position: 'insideBottomRight', offset: -5 }}
                  />
                  <YAxis
                    domain={[0, 120]}
                    label={{ value: 'Response', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    content={({ active, payload }) => {
                      if (!active || !payload || payload.length === 0) return null;
                      
                      // Get the dose (x value) from the first payload entry
                      const dose = payload[0]?.payload?.x;
                      
                      // Find model fit and confidence limits
                      const modelFit = payload.find(p => p.name === "Model Fit");
                      const upper = payload[0]?.payload?.upper;
                      const lower = payload[0]?.payload?.lower;
                      
                      return (
                        <div className="bg-white border border-gray-300 rounded p-2 shadow-lg">
                          <div className="text-sm font-semibold mb-1">
                            Dose: {fmt(dose)} mg/kg
                          </div>
                          {modelFit && (
                            <div className="text-sm">
                              <span style={{ color: modelFit.color }}>Prediction: </span>
                              <span className="font-semibold">{fmt(modelFit.value)}</span>
                            </div>
                          )}
                          {upper !== undefined && lower !== undefined && (
                            <div className="text-sm text-gray-600">
                              90% CI: [{fmt(lower)}, {fmt(upper)}]
                            </div>
                          )}
                        </div>
                      );
                    }}
                  />
                  <Legend verticalAlign="top" height={36} />
                  
                  {/* Dose reference lines - more distinct */}
                  {doses.map((dose, idx) => (
                    <ReferenceLine
                      key={idx}
                      x={dose}
                      stroke="#1e293b"
                      strokeWidth={2}
                      label={null}
                    />
                  ))}
                  
                  {/* Confidence band (Expected Precision) */}
                  {confidenceBands && precisionMetrics && (
                    <Area
                      name="Expected Precision"
                      data={confidenceBands}
                      dataKey="upper"
                      fill={precisionMetrics.color}
                      fillOpacity={0.3}
                      stroke="none"
                      isAnimationActive={false}
                      legendType="rect"
                    />
                  )}
                  {confidenceBands && (
                    <Area
                      data={confidenceBands}
                      dataKey="lower"
                      fill="#ffffff"
                      fillOpacity={1}
                      stroke="none"
                      isAnimationActive={false}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">How to use:</h3>
          <ol className="list-decimal list-inside space-y-1 text-sm text-blue-800">
            <li>Enter the dose levels you're planning to test (e.g., 0.5, 1, 2, 4)</li>
            <li>The colored band shows how well your data will support precise modeling</li>
            <li><strong>Blue band = good precision</strong> (modeling will produce precise predictions)</li>
            <li><strong>Red band = poor precision</strong> (model predictions will be uncertain)</li>
            <li>Try different dose combinations to see which designs give the best precision</li>
          </ol>
          <p className="text-xs text-blue-700 mt-2">
            <strong>Goal:</strong> Find a dose design where the colored band is narrow and blue. This means your experiment will produce high-quality data!
          </p>
        </div>
      </div>
    </div>
  );
}
