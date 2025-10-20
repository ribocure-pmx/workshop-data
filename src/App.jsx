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
  Scatter,
  ResponsiveContainer,
} from 'recharts';

// Emax model function with H=1
function emaxModel(dose, E0, Emax, ED50) {
  const h = 1; // Fixed Hill coefficient
  const num = Emax * Math.pow(dose, h);
  const den = Math.pow(ED50, h) + Math.pow(dose, h);
  return E0 - (den > 0 ? num / den : 0);
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

// Calculate residual sum of squares
function calculateRSS(observed, predicted) {
  return observed.reduce((sum, obs, i) => {
    const diff = obs.y - predicted[i];
    return sum + diff * diff;
  }, 0);
}

// Simple gradient descent for parameter optimization
function optimizeParameters(data, initialParams, iterations = 1000, learningRate = 0.01) {
  const E0 = 100; // Fixed E0
  let { Emax, ED50 } = initialParams;
  const bestParams = { Emax, ED50 };
  let bestRSS = Infinity;

  for (let iter = 0; iter < iterations; iter++) {
    const predicted = data.map(d => emaxModel(d.x, E0, Emax, ED50));
    const rss = calculateRSS(data, predicted);

    if (rss < bestRSS) {
      bestRSS = rss;
      bestParams.Emax = Emax;
      bestParams.ED50 = ED50;
    }

    // Calculate gradients with finite differences
    const delta = 0.001;
    const gradEmax = (calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax + delta, ED50))) - rss) / delta;
    const gradED50 = (calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax, ED50 + delta))) - rss) / delta;

    // Update parameters
    Emax -= learningRate * gradEmax;
    ED50 -= learningRate * gradED50;

    // Keep parameters in reasonable bounds
    Emax = Math.max(0, Math.min(100, Emax));
    ED50 = Math.max(0.01, Math.min(10, ED50));

    // Reduce learning rate over time
    if (iter % 100 === 0) {
      learningRate *= 0.95;
    }
  }

  return bestParams;
}

// Calculate variance-covariance matrix and standard errors using Hessian matrix (2nd derivatives)
// Returns both the standard errors and the full covariance matrix for multivariate uncertainty
function calculateCovarianceMatrix(data, params) {
  const E0 = 100; // Fixed E0
  const { Emax, ED50 } = params;
  
  // Calculate residual variance (sigma^2)
  const predicted = data.map(d => emaxModel(d.x, E0, Emax, ED50));
  const residuals = data.map((d, i) => d.y - predicted[i]);
  const rss = residuals.reduce((sum, r) => sum + r * r, 0);
  const sigma2 = rss / (data.length - 2); // 2 parameters (Emax, ED50)
  
  // Compute Hessian matrix (2x2) using finite differences
  const delta = 0.001;
  
  // Second derivatives
  const d2RSS_dEmax2 = (
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax + delta, ED50))) +
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax - delta, ED50))) -
    2 * rss
  ) / (delta * delta);
  
  const d2RSS_dED502 = (
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax, ED50 + delta))) +
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax, ED50 - delta))) -
    2 * rss
  ) / (delta * delta);
  
  const d2RSS_dEmaxdED50 = (
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax + delta, ED50 + delta))) -
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax + delta, ED50 - delta))) -
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax - delta, ED50 + delta))) +
    calculateRSS(data, data.map(d => emaxModel(d.x, E0, Emax - delta, ED50 - delta)))
  ) / (4 * delta * delta);
  
  // Hessian matrix H = [[H11, H12], [H21, H22]]
  const H11 = d2RSS_dEmax2 / 2;
  const H22 = d2RSS_dED502 / 2;
  const H12 = d2RSS_dEmaxdED50 / 2;
  
  // Invert 2x2 Hessian: inv(H) = 1/det * [[H22, -H12], [-H12, H11]]
  const det = H11 * H22 - H12 * H12;
  
  if (det <= 0) {
    // Fallback if Hessian is singular or not positive definite
    console.warn('Hessian not positive definite, using approximate covariance');
    return {
      standardErrors: {
        Emax: Math.sqrt(Math.abs(sigma2 / H11)) || 1,
        ED50: Math.sqrt(Math.abs(sigma2 / H22)) || 0.1
      },
      covMatrix: {
        var_Emax: Math.abs(sigma2 / H11),
        var_ED50: Math.abs(sigma2 / H22),
        cov_Emax_ED50: 0
      }
    };
  }
  
  // Variance-covariance matrix = sigma^2 * inv(H)
  const var_Emax = sigma2 * H22 / det;
  const var_ED50 = sigma2 * H11 / det;
  const cov_Emax_ED50 = -sigma2 * H12 / det;
  
  return {
    standardErrors: {
      Emax: Math.sqrt(Math.abs(var_Emax)),
      ED50: Math.sqrt(Math.abs(var_ED50))
    },
    covMatrix: {
      var_Emax,
      var_ED50,
      cov_Emax_ED50
    }
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
  // True underlying parameters for simulation (E0 fixed at 100)
  const [trueParams] = useState({ E0: 100, Emax: 80, ED50: 1.5 });
  
  // User-adjusted parameters (E0 fixed at 100)
  const [params, setParams] = useState({ Emax: 60, ED50: 2 });
  
  const E0 = 100; // Fixed baseline
  
  // Dose input
  const [doseInput, setDoseInput] = useState('0.5, 1, 2, 3, 4');
  
  // Simulated experimental data
  const [experimentData, setExperimentData] = useState([]);
  
  // Optimized parameters, their standard errors, and covariance matrix
  const [optimizedParams, setOptimizedParams] = useState(null);
  const [standardErrors, setStandardErrors] = useState(null);
  const [covMatrix, setCovMatrix] = useState(null);
  const [isOptimizing, setIsOptimizing] = useState(false);

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

  // Run experiment - simulate data with noise
  const runExperiment = () => {
    if (doses.length === 0) {
      alert('Please enter valid dose values (comma-separated)');
      return;
    }

    const data = doses.map(dose => ({
      x: dose,
      y: addNoise(emaxModel(dose, trueParams.E0, trueParams.Emax, trueParams.ED50), 5)
    }));

    setExperimentData(data);
    setOptimizedParams(null);
    setStandardErrors(null);
    setCovMatrix(null);
  };

  // Optimize parameters
  const optimizeParams = async () => {
    if (experimentData.length === 0) {
      alert('Please run an experiment first');
      return;
    }

    setIsOptimizing(true);
    
    // Use setTimeout to allow UI to update
    setTimeout(() => {
      const optimized = optimizeParameters(experimentData, params, 2000, 0.05);
      const result = calculateCovarianceMatrix(experimentData, optimized);
      
      setOptimizedParams(optimized);
      setStandardErrors(result.standardErrors);
      setCovMatrix(result.covMatrix);
      setParams(optimized); // Update sliders to optimized values
      setIsOptimizing(false);
    }, 100);
  };

  // Generate smooth curve for current parameters (dense for smooth hovering)
  const curve = useMemo(() => {
    const maxDose = Math.max(...doses, 5);
    const points = Array.from({ length: 201 }, (_, i) => (i / 200) * maxDose);
    return points.map(x => ({
      x,
      y: Math.max(0, Math.min(120, emaxModel(x, E0, params.Emax, params.ED50)))
    }));
  }, [params, doses]);

  // Generate confidence bands using multivariate sampling
  const confidenceBands = useMemo(() => {
    if (!optimizedParams || !covMatrix) return null;
    
    const maxDose = Math.max(...doses, 5);
    const points = Array.from({ length: 201 }, (_, i) => (i / 200) * maxDose);
    
    // Sample from multivariate normal distribution
    const samples = sampleMultivariateNormal(optimizedParams, covMatrix, 200);
    
    if (samples.length === 0) return null;
    
    return points.map(x => {
      const yFit = emaxModel(x, E0, optimizedParams.Emax, optimizedParams.ED50);
      
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
  }, [optimizedParams, covMatrix, doses]);

  // Calculate RSS if we have experiment data
  const rss = useMemo(() => {
    if (experimentData.length === 0) return null;
    const predicted = experimentData.map(d => emaxModel(d.x, E0, params.Emax, params.ED50));
    return calculateRSS(experimentData, predicted);
  }, [experimentData, params]);

  return (
    <div className="min-h-screen w-full bg-white text-gray-900">
      <div className="max-w-6xl mx-auto p-6">
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Interactive Emax Model Fitting</h1>
          <p className="text-gray-600 mt-2">
            Simulate experiments and fit an Emax pharmacodynamic model (H = 1)
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Left panel - Controls */}
          <div className="md:col-span-1 space-y-4">
            {/* Dose Input */}
            <div className="bg-gray-50 rounded-2xl p-4 shadow-sm border border-gray-200">
              <h3 className="font-semibold mb-2">Dose Input</h3>
              <input
                type="text"
                value={doseInput}
                onChange={(e) => setDoseInput(e.target.value)}
                placeholder="e.g., 0.5, 1, 2, 3, 4"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              <p className="text-xs text-gray-500 mt-1">Enter comma-separated dose values (mg/kg)</p>
              <button
                onClick={runExperiment}
                className="mt-3 w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium"
              >
                Run Experiment
              </button>
            </div>

            {/* Parameter Sliders */}
            <div className="bg-gray-50 rounded-2xl p-4 shadow-sm border border-gray-200">
              <h3 className="font-semibold mb-3">Model Parameters</h3>
              
              <div className="mb-3 text-sm bg-white border border-gray-200 rounded-xl p-3">
                <div className="font-medium mb-1">Equation (H = 1, E0 = 100)</div>
                <div className="font-mono text-xs">
                  y = 100 − (Emax·x)/(ED50 + x)
                </div>
              </div>

              <div className="space-y-4">
                <SliderRow
                  label="Emax"
                  min={0}
                  max={100}
                  step={0.5}
                  value={params.Emax}
                  onChange={(v) => updateParam('Emax', v)}
                />
                <SliderRow
                  label="ED50"
                  min={0.1}
                  max={10}
                  step={0.1}
                  value={params.ED50}
                  onChange={(v) => updateParam('ED50', v)}
                />
              </div>

              {rss !== null && (
                <div className="mt-4 pt-3 border-t border-gray-300">
                  <div className="text-xs uppercase tracking-wide text-gray-500">RSS</div>
                  <div className="text-base font-semibold">{fmt(rss, 1)}</div>
                </div>
              )}
            </div>

            {/* Optimize Button */}
            {experimentData.length > 0 && (
              <div className="bg-gray-50 rounded-2xl p-4 shadow-sm border border-gray-200">
                <button
                  onClick={optimizeParams}
                  disabled={isOptimizing}
                  className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {isOptimizing ? 'Optimizing...' : 'Optimize Parameters'}
                </button>

                {optimizedParams && standardErrors && (
                  <div className="mt-4 space-y-2">
                    <h4 className="font-semibold text-sm">Optimal Parameters:</h4>
                    <div className="text-sm space-y-1 bg-white rounded-lg p-3 border border-gray-200">
                      <div className="flex justify-between">
                        <span>E0:</span>
                        <span className="font-mono">100 (fixed)</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Emax:</span>
                        <span className="font-mono">{fmt(optimizedParams.Emax)} ± {fmt(standardErrors.Emax)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>ED50:</span>
                        <span className="font-mono">{fmt(optimizedParams.ED50)} ± {fmt(standardErrors.ED50)}</span>
                      </div>
                      {covMatrix && (
                        <div className="flex justify-between text-xs pt-2 border-t border-gray-200 mt-2">
                          <span>Correlation:</span>
                          <span className="font-mono">
                            {fmt(covMatrix.cov_Emax_ED50 / Math.sqrt(covMatrix.var_Emax * covMatrix.var_ED50), 3)}
                          </span>
                        </div>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      Confidence band uses multivariate uncertainty (accounts for parameter correlation)
                    </p>
                  </div>
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
                      
                      // Filter out "Observed Data" from payload
                      const filteredPayload = payload.filter(p => p.name !== "Observed Data");
                      if (filteredPayload.length === 0) return null;
                      
                      // Get the dose (x value) from the first payload entry
                      const dose = payload[0]?.payload?.x;
                      
                      // Find model fit and confidence limits
                      const modelFit = filteredPayload.find(p => p.name === "Model Fit");
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
                  
                  {/* Confidence band (90% CI) */}
                  {confidenceBands && (
                    <Area
                      name="90% CI"
                      data={confidenceBands}
                      dataKey="upper"
                      fill="#8884d8"
                      fillOpacity={0.2}
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
                  
                  {/* Model curve - uses its own data for smooth hovering */}
                  <Line
                    name="Model Fit"
                    data={confidenceBands || curve}
                    type="monotone"
                    dataKey="y"
                    stroke="#8884d8"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                  
                  {/* Experimental data points - visible but filtered from tooltip */}
                  {experimentData.length > 0 && (
                    <Scatter
                      name="Observed Data"
                      data={experimentData}
                      dataKey="y"
                      fill="#ef4444"
                      stroke="#dc2626"
                      strokeWidth={2}
                      isAnimationActive={false}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {experimentData.length === 0 && (
              <div className="text-center text-gray-500 mt-4">
                <p>Enter doses and click "Run Experiment" to generate data</p>
              </div>
            )}
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">How to use:</h3>
          <ol className="list-decimal list-inside space-y-1 text-sm text-blue-800">
            <li>Enter dose values (comma-separated) in the text box</li>
            <li>Click "Run Experiment" to simulate data from the underlying Emax model with noise</li>
            <li>Use the sliders to manually adjust parameters and visualize the fit</li>
            <li>Click "Optimize Parameters" to automatically find the best-fit parameters</li>
            <li>View the optimal parameters with their standard errors</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
