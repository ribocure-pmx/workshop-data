# Interactive Emax Model Fitting

A React + Vite application for interactive Emax pharmacodynamic model fitting with data simulation and parameter estimation.

## Features

- **Dose Input**: Enter custom dose values (comma-separated) for your experiment
- **Data Simulation**: Generate experimental data with realistic noise from an underlying Emax model
- **Interactive Parameter Sliders**: Manually adjust E0, Emax, and ED50 parameters and see real-time fit
- **Automatic Optimization**: Find optimal parameters using gradient descent optimization
- **Standard Errors**: View parameter estimates with calculated standard errors
- **Real-time Visualization**: See your model fit and data on an interactive chart

## Model

The app uses a simplified Emax model with Hill coefficient fixed at H=1:

```
y = E0 − (Emax·x)/(ED50 + x)
```

Where:
- **E0**: Baseline response (when dose = 0)
- **Emax**: Maximum effect/drop from baseline
- **ED50**: Dose that produces 50% of maximum effect
- **x**: Dose (mg/kg)

## Prerequisites

- Node.js 18 or newer ([https://nodejs.org/](https://nodejs.org/))
- npm (bundled with Node.js)

## Installation

```powershell
# Install dependencies
npm install
```

## Usage

```powershell
# Start development server
npm run dev
```

Then open your browser to the URL shown in the terminal (usually http://localhost:5173)

## How to Use the App

1. **Enter Doses**: Type comma-separated dose values in the input box (e.g., "0.5, 1, 2, 3, 4")
2. **Run Experiment**: Click "Run Experiment" to simulate data with noise
3. **Adjust Parameters**: Use sliders to manually fit the model to the data
4. **Optimize**: Click "Optimize Parameters" for automatic parameter estimation
5. **View Results**: See optimal parameters with standard errors

## Build for Production

```powershell
npm run build
npm run preview
```

## Technologies

- **React 18**: UI library
- **Vite**: Fast build tool
- **Recharts**: Data visualization
- **Tailwind CSS**: Styling

## Project Structure

```
├── src/
│   ├── App.jsx        # Main application component
│   ├── main.jsx       # React entry point
│   └── index.css      # Tailwind CSS styles
├── index.html         # HTML template
├── package.json       # Dependencies and scripts
├── vite.config.js     # Vite configuration
└── tailwind.config.js # Tailwind configuration
```

## License

MIT
