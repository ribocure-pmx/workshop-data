import React, { useMemo, useState } from 'react';
import {
  ComposedChart,
  Line,
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
function addNoise(value, noiseLevel = 5) {
  return value + (Math.random() - 0.5) * 2 * noiseLevel;
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

// Calculate standard errors using bootstrap-like approach
function calculateStandardErrors(data, params, numSamples = 100) {
  const E0 = 100; // Fixed E0
  const estimates = { Emax: [], ED50: [] };

  for (let i = 0; i < numSamples; i++) {
    // Resample data with noise
    const resampledData = data.map(d => ({
      x: d.x,
      y: addNoise(emaxModel(d.x, E0, params.Emax, params.ED50), 3)
    }));

    const optimized = optimizeParameters(resampledData, params, 200, 0.02);
    estimates.Emax.push(optimized.Emax);
    estimates.ED50.push(optimized.ED50);
  }

  // Calculate standard deviation of estimates
  const calcSD = (arr) => {
    const mean = arr.reduce((a, b) => a + b) / arr.length;
    const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
  };

  return {
    Emax: calcSD(estimates.Emax),
    ED50: calcSD(estimates.ED50)
  };
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
  
  // Optimized parameters and their standard errors
  const [optimizedParams, setOptimizedParams] = useState(null);
  const [standardErrors, setStandardErrors] = useState(null);
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
      const se = calculateStandardErrors(experimentData, optimized, 50);
      
      setOptimizedParams(optimized);
      setStandardErrors(se);
      setParams(optimized); // Update sliders to optimized values
      setIsOptimizing(false);
    }, 100);
  };

  // Generate smooth curve for current parameters
  const curve = useMemo(() => {
    const maxDose = Math.max(...doses, 5);
    const points = Array.from({ length: 200 }, (_, i) => (i / 199) * maxDose);
    return points.map(x => ({
      x,
      y: Math.max(0, Math.min(120, emaxModel(x, E0, params.Emax, params.ED50)))
    }));
  }, [params, doses]);

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
                  label="Max Effect Emax"
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
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      Values shown as estimate ± standard error
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
                  <Tooltip formatter={(val) => fmt(val)} />
                  <Legend verticalAlign="top" height={36} />
                  
                  {/* Model curve */}
                  <Line
                    name="Model Fit"
                    type="monotone"
                    dataKey="y"
                    stroke="#8884d8"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                  
                  {/* Experimental data points */}
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
