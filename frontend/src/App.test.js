import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('axios');

test('renders Clinician Guided Grad-CAM heading', () => {
  render(<App />);
  const headingElement = screen.getByText(/Clinician Guided Grad-CAM for Tuberculosis Detection/i);
  expect(headingElement).toBeInTheDocument();
});

test('renders TB label options', () => {
  render(<App />);
  const tbPresentLabel = screen.getByText(/TB Present/i);
  const normalLabel = screen.getByText(/Normal/i);
  expect(tbPresentLabel).toBeInTheDocument();
  expect(normalLabel).toBeInTheDocument();
});

test('renders Model Management section', () => {
  render(<App />);
  const modelManagementHeading = screen.getByText(/Model Management/i);
  expect(modelManagementHeading).toBeInTheDocument();
});
