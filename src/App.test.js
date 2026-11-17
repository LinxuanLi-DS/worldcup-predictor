import { render, screen } from '@testing-library/react';
import App from './App';
import { SCHEDULE } from './data';

const originalFetch = global.fetch;

beforeEach(() => {
  global.fetch = jest.fn().mockResolvedValue({
    json: async () => ({ events: [] }),
  });
});

afterEach(() => {
  global.fetch = originalFetch;
});

test('shows the fixture schedule and labels the completed-match metric as a backtest', () => {
  render(<App />);

  const completedCount = SCHEDULE.filter(match => match.completed).length;
  expect(screen.getByRole('heading', { name: /赛果\s*预测\s*系统/ })).toBeInTheDocument();
  expect(screen.getByText(`已赛 ${completedCount} 场 · 回测准确率`)).toBeInTheDocument();
  expect(screen.getAllByText(/预测比分/).length).toBeGreaterThan(0);
});
