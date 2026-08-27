# Transport and Grasp Component Analysis

This repository contains the R analysis code for my MSc dissertation at the University of Leeds, supervised by Dr Carlo Campagnoli and Dr Elham Mirfarah.

## Overview

The dissertation investigates how visual information about object size influences the two components of reach-to-grasp movements: the transport component (hand path to the object) and the grasp component (finger aperture scaling). Participants reached to objects placed at varying offsets from a visual reference, under conditions designed to dissociate visual and haptic size information.

Two experiments were conducted. Experiment 1 used a blocked design with 10 participants. Experiment 2 used a fully randomised design with 13 participants, providing a stronger test of trial-by-trial visual updating.

## Analysis

Classification models — Lasso, SVM (RBF kernel), Random Forest, and FDA with functional principal component scores — were trained to predict object offset category from kinematic features of each movement component. Model performance was evaluated using Leave-One-Participant-Out Cross-Validation (LOPO-CV) to assess generalisation across participants.

Key features extracted include maximum grip aperture, time to maximum grip aperture, peak velocity, time to peak velocity, and movement duration, alongside functional data analysis summaries of the full kinematic trajectory.

## Repository Contents

- `dissertation_rcode.R` — full analysis pipeline from raw data import through feature extraction, model fitting, LOPO-CV, and visualisation

## Notes

Raw data are not included in this repository in line with data governance requirements. Code is shared for transparency and reproducibility of the analytical approach.
