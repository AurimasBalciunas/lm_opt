import matplotlib.pyplot as plt
import numpy as np


BEACONS = np.array([[0.0, 0.0], [6.0, 0.0], [1.0, 5.0]])
TRUE_POSITION = np.array([2.0, 2.0])
INITIAL_GUESS = np.array([4.0, 4.0])

# Fixed measurement errors
NOISE = np.array([0.15, -0.10, 0.20])
EXACT_DISTANCES = np.linalg.norm(TRUE_POSITION - BEACONS, axis=1)
MEASURED_DISTANCES = EXACT_DISTANCES + NOISE
LAMBDA = 0.1


def residuals(position, beacons, measured_distances):
    """Return predicted distance minus measured distance for each beacon."""
    predicted_distances = []
    for i in range(len(beacons)):
      dx = position[0] - beacons[i][0]
      dy = position[1] - beacons[i][1]
      predicted_distances.append(np.sqrt(dx**2 + dy**2))
    return np.array(predicted_distances) - measured_distances
    
    
def jacobian(position, beacons):
  """Jacobian denotes how changing our position slightly affects our residual"""
  # residual is predicted distance - measured distance
  # using chain rule,
  # ∂ri / ∂x = 1/2(predicted_distance)  * 2(px - bix) = (px - bix) / predicted_distance
  # predicted distance in this case is np.sqrt(dx**2 + dy**2)
  # jacobian is going to be one [∂ri/∂x  ∂ri/∂y] row per beacon
  J = np.zeros((len(beacons), 2))
  for i in range(len(beacons)):
    dx = position[0] - beacons[i][0]
    dy = position[1] - beacons[i][1]
    predicted_distance = np.sqrt(dx**2 + dy**2)
    
    if predicted_distance == 0:
      raise ValueError("Jacobian is undefined at beacon position")

    dridx = dx / predicted_distance
    dridy = dy / predicted_distance
    J[i] = [dridx, dridy]

  return J

def lm_step(position, beacons, measured_distances, damping):
    r = residuals(position, beacons, measured_distances)
    J = jacobian(position, beacons)

    # A = JᵀJ + damping * I
    A = J.T@J + damping * np.identity(2)
    # b = -Jᵀr
    b = -J.T@r
    # Solve A @ delta = b and return delta
    delta = np.linalg.solve(A, b)
    return delta

def lm_opt(position, beacons, measured_distances, damping):
  # 1. current cost
  r_before = residuals(position, beacons, measured_distances)
  cost_before = 0.5 * (r_before @ r_before)
  # 2. propose delta with lm step
  delta = lm_step(position, beacons, measured_distances, damping)
  # 3. calc cost at position + delta
  trial_position = position + delta
  r_after = residuals(trial_position, beacons, measured_distances)
  cost_after = 0.5 * (r_after @ r_after)
  
  # 4. accept or reject proposal (accept -> move and damping /= 2, deny -> don't move and damping *= 2)
  accepted = cost_after < cost_before
  if accepted:
    return trial_position, damping/2, accepted
  else:
    return position, damping*2, accepted

def plot_problem(path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (beacon, distance) in enumerate(zip(BEACONS, MEASURED_DISTANCES)):
        color = f"C{i}"
        ax.scatter(*beacon, color=color, marker="^", s=90)
        ax.annotate(f"Beacon {i}", beacon, xytext=(6, 6), textcoords="offset points")
        ax.add_patch(plt.Circle(beacon, distance, fill=False, color=color, alpha=0.65))
    ax.scatter(*TRUE_POSITION, color="black", marker="*", s=140, label="True position")
    ax.scatter(*INITIAL_GUESS, color="purple", marker="x", s=90, label="Initial guess")
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o--", color="purple", markersize=4,
            label="Accepted LM steps")
    ax.scatter(*path[-1], color="red", marker="X", s=100, label="Estimated position")
    ax.set(xlim=(-5, 11), ylim=(-5, 10), xlabel="x", ylabel="y",
           title="Each circle represents one noisy distance measurement")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig("result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    position = INITIAL_GUESS.copy()
    damping = LAMBDA
    path = [position.copy()]
    max_iterations = 100
    gradient_tolerance = 1e-8
    stop_reason = "Maximum iterations reached"

    r = residuals(position, BEACONS, MEASURED_DISTANCES)
    print(f"Initial position: {position}, cost: {0.5 * (r @ r):.8f}")
    print("Attempt  Cost          Damping for next attempt  Accepted")

    for iteration in range(max_iterations):
        r = residuals(position, BEACONS, MEASURED_DISTANCES)
        J = jacobian(position, BEACONS)
        if np.linalg.norm(J.T @ r, ord=np.inf) < gradient_tolerance:
            stop_reason = "Converged: cost gradient is small"
            break

        position, damping, accepted = lm_opt(
            position, BEACONS, MEASURED_DISTANCES, damping
        )
        if accepted:
            path.append(position.copy())

        r = residuals(position, BEACONS, MEASURED_DISTANCES)
        print(f"{iteration + 1:7d}  {0.5 * (r @ r):.8f}  {damping:24.3g}  {accepted}")

    print(stop_reason)
    print(f"Estimated position: {position}")
    print(f"True position:      {TRUE_POSITION}")
    print(f"Final cost: {0.5 * (r @ r):.8f}")
    print("Plot saved to result.png")
    plot_problem(path)
