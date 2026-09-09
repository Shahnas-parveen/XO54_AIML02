# Adaptive Hospital Demand Forecasting

## X'O CODE 2026 | PS02 – Adaptive Time-Series Forecasting Under Changing Conditions

### Overview

Adaptive Hospital Demand Forecasting is an AI/ML-based time-series forecasting system designed to predict the next day's hospital patient demand and adapt its forecasting model when new actual observations become available.

Unlike a static forecasting system that is trained once and used continuously, this system follows an iterative forecasting workflow:

Historical Patient Data → Forecast → Actual Observation → Error Evaluation → Model Update → New Forecast

The goal is to support hospital operational planning by providing an updated estimate of expected patient demand while responding to changes in demand patterns.

---

## Problem Statement

Real-world demand patterns can change because of factors such as changing patient behaviour, seasonal conditions, policies, weather, supply-chain effects, or unexpected events.

A forecasting model trained on historical data may therefore become less reliable when the underlying pattern changes.

This project addresses the problem of **Adaptive Time-Series Forecasting Under Changing Conditions** by continuously evaluating forecast performance and updating the forecasting model when new observations become available.

The system is designed to avoid relying on manually specified change dates and to support adaptation based on observed forecasting behaviour.

---

## Proposed Solution

The system uses historical patient-count time-series data to train a machine learning forecasting model.

For V1, the forecasting process uses the recent history of patient counts to predict the next day's demand.

### Forecasting Workflow

```text
Historical Patient Data
          ↓
   Train Forecast Model
          ↓
   Predict Next-Day Demand
          ↓
     Actual Count Arrives
          ↓
    Calculate Forecast Error
          ↓
     Update Training Data
          ↓
       Retrain Model
          ↓
   Predict Following Day
          ↓
         Repeat