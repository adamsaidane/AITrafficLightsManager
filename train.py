"""
Q-Learning Amélioré pour Contrôle des Feux de Circulation
Avec visualisations, sauvegarde, et métriques avancées
"""

import traci
import sumolib
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration centralisée"""

    # Simulation
    TL_ID = "center"
    MIN_GREEN_TIME = 10
    MAX_GREEN_TIME = 60
    YELLOW_TIME = 3
    TOTAL_STEPS = 3600
    NUM_EPISODES = 100

    # Q-Learning avec les 17 phases compatibles
    ALPHA = 0.1  # Learning rate
    GAMMA = 0.95  # Discount factor (augmenté pour vision long terme)
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.995  # Décroissance plus agressive
    EPSILON_MIN = 0.01

    # Récompenses
    REWARD_WAIT_MULTIPLIER = -0.1
    REWARD_THROUGHPUT_MULTIPLIER = 1.0
    REWARD_CHANGE_PENALTY = -5  # Pénalité changement de phase

    # Les 17 phases compatibles (format: action_id -> state_string)
    PHASES = {
        0: 'GggrGrrrGggrGrrr',  # Nord-Sud droit
        1: 'GrrrGggrGrrrGggr',  # Est-Ouest droit
        2: 'GggrGrrgGrrrGrrr',  # Nord droit + Sud gauche
        3: 'GrrgGggrGrrrGrrr',  # Sud droit + Nord gauche
        4: 'GrrrGrrrGggrGrrg',  # Est droit + Ouest gauche
        5: 'GrrrGrrrGrrgGggr',  # Ouest droit + Est gauche
        6: 'GrrgGrrrGrrgGrrr',  # Nord+Sud gauche
        7: 'GrrrGrrgGrrrGrrg',  # Est+Ouest gauche
    }

    NUM_ACTIONS = len(PHASES)

    # Voies à surveiller
    LANES = [
        'N_to_C_0', 'N_to_C_1', 'N_to_C_2', 'N_to_C_3',
        'S_to_C_0', 'S_to_C_1', 'S_to_C_2', 'S_to_C_3',
        'E_to_C_0', 'E_to_C_1', 'E_to_C_2', 'E_to_C_3',
        'W_to_C_0', 'W_to_C_1', 'W_to_C_2', 'W_to_C_3'
    ]

    # Dossiers
    MODELS_DIR = 'models'
    PLOTS_DIR = 'plots'


# ==============================================================================
# CLASSE AGENT Q-LEARNING AMÉLIORÉ
# ==============================================================================

class ImprovedQLearningAgent:
    """Agent Q-Learning avec état discréti amélioré"""

    def __init__(self, config):
        self.config = config

        # Discrétisation de l'état:
        # - Pression trafic: 5 niveaux (NS_High, NS_Low, Balanced, EW_Low, EW_High)
        # - Phase actuelle: 17 phases
        # Total: 5 × 17 = 85 états
        self.num_pressure_levels = 5
        self.num_states = self.num_pressure_levels * config.NUM_ACTIONS

        # Table Q
        self.q_table = np.zeros((self.num_states, config.NUM_ACTIONS))

        # Paramètres d'apprentissage
        self.alpha = config.ALPHA
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN

        # Historique
        self.episode_rewards = []
        self.episode_waiting_times = []
        self.episode_throughputs = []
        self.epsilon_history = []

        # Métriques par épisode
        self.current_episode_stats = {
            'total_reward': 0,
            'total_waiting_time': 0,
            'vehicles_passed': 0,
            'phase_changes': 0
        }

    def _discretize_pressure(self):
        """
        Discrétise la pression du trafic en 5 niveaux
        Retourne: 0-4 (NS_High, NS_Med, Balanced, EW_Med, EW_High)
        """
        try:
            # Calculer pression Nord-Sud vs Est-Ouest
            ns_lanes = [l for l in self.config.LANES if l.startswith(('N_', 'S_'))]
            ew_lanes = [l for l in self.config.LANES if l.startswith(('E_', 'W_'))]

            ns_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane)
                              for lane in ns_lanes if lane in traci.lane.getIDList())
            ew_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane)
                              for lane in ew_lanes if lane in traci.lane.getIDList())

            # Différence de pression
            pressure_diff = ns_vehicles - ew_vehicles

            # Discrétisation en 5 niveaux
            if pressure_diff > 10:
                return 0  # NS très chargé
            elif pressure_diff > 3:
                return 1  # NS moyennement chargé
            elif abs(pressure_diff) <= 3:
                return 2  # Équilibré
            elif pressure_diff < -3:
                return 3  # EW moyennement chargé
            else:
                return 4  # EW très chargé

        except:
            return 2  # Équilibré par défaut

    def get_state(self, current_phase):
        """Retourne l'index de l'état actuel"""
        pressure_level = self._discretize_pressure()
        return pressure_level * self.config.NUM_ACTIONS + current_phase

    def choose_action(self, state):
        """Politique epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.config.NUM_ACTIONS - 1)
        else:
            # Choisir la meilleure action
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            # Si plusieurs actions ont la même valeur Q maximale, choisir aléatoirement
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        """Mise à jour Q-Learning"""
        old_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])

        # Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        """Décroissance epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_episode_stats(self, reward, waiting_time, throughput):
        """Met à jour les statistiques de l'épisode"""
        self.current_episode_stats['total_reward'] += reward
        self.current_episode_stats['total_waiting_time'] += waiting_time
        self.current_episode_stats['vehicles_passed'] = throughput

    def finish_episode(self):
        """Finalise l'épisode et sauvegarde les stats"""
        self.episode_rewards.append(self.current_episode_stats['total_reward'])
        self.episode_waiting_times.append(self.current_episode_stats['total_waiting_time'])
        self.episode_throughputs.append(self.current_episode_stats['vehicles_passed'])
        self.epsilon_history.append(self.epsilon)

        # Reset stats
        self.current_episode_stats = {
            'total_reward': 0,
            'total_waiting_time': 0,
            'vehicles_passed': 0,
            'phase_changes': 0
        }

    def save_model(self, filepath):
        """Sauvegarde la table Q et les paramètres"""
        data = {
            'q_table': self.q_table.tolist(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_waiting_times': self.episode_waiting_times,
            'episode_throughputs': self.episode_throughputs,
            'epsilon_history': self.epsilon_history,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Modèle sauvegardé: {filepath}")

    def load_model(self, filepath):
        """Charge la table Q et les paramètres"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.q_table = np.array(data['q_table'])
        self.epsilon = data['epsilon']
        self.episode_rewards = data['episode_rewards']
        self.episode_waiting_times = data['episode_waiting_times']
        self.episode_throughputs = data['episode_throughputs']
        self.epsilon_history = data['epsilon_history']

        print(f"✅ Modèle chargé: {filepath}")


# ==============================================================================
# ENVIRONNEMENT DE SIMULATION
# ==============================================================================

class SUMOEnvironment:
    """Environnement de simulation SUMO"""

    def __init__(self, config, use_gui=False):
        self.config = config
        self.use_gui = use_gui
        self.current_phase = 0
        self.time_on_phase = 0
        self.is_yellow = False

    def start(self):
        """Démarre SUMO"""
        sumoBinary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')

        traci.start([
            sumoBinary,
            "-c", "intersection.sumocfg",
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--waiting-time-memory", "1000"
        ])

        # Définir phase initiale
        traci.trafficlight.setRedYellowGreenState(
            self.config.TL_ID,
            self.config.PHASES[self.current_phase]
        )

    def step(self):
        """Fait un pas de simulation"""
        traci.simulationStep()
        self.time_on_phase += 1

    def calculate_reward(self, phase_changed=False):
        """
        Calcule la récompense basée sur:
        - Temps d'attente (pénalité)
        - Débit (bonus)
        - Changement de phase (petite pénalité)
        """
        # Temps d'attente total
        total_waiting = 0
        for lane in self.config.LANES:
            if lane in traci.lane.getIDList():
                total_waiting += traci.lane.getWaitingTime(lane)

        # Nombre de véhicules arrivés
        arrived = traci.simulation.getArrivedNumber()

        # Calculer récompense
        reward = (self.config.REWARD_WAIT_MULTIPLIER * total_waiting +
                  self.config.REWARD_THROUGHPUT_MULTIPLIER * arrived)

        # Pénalité pour changement de phase
        if phase_changed:
            reward += self.config.REWARD_CHANGE_PENALTY

        return reward, total_waiting, arrived

    def can_change_phase(self):
        """Vérifie si on peut changer de phase"""
        return (not self.is_yellow and
                self.time_on_phase >= self.config.MIN_GREEN_TIME)

    def change_phase(self, new_phase):
        """Change la phase du feu"""
        if new_phase != self.current_phase:
            # Phase jaune
            yellow_state = self.config.PHASES[self.current_phase].replace('g', 'y').replace('G', 'y')
            traci.trafficlight.setRedYellowGreenState(self.config.TL_ID, yellow_state)
            self.is_yellow = True
            self.time_on_phase = 0
            return new_phase
        return self.current_phase

    def apply_phase(self, phase):
        """Applique une phase verte"""
        traci.trafficlight.setRedYellowGreenState(
            self.config.TL_ID,
            self.config.PHASES[phase]
        )
        self.current_phase = phase
        self.is_yellow = False
        self.time_on_phase = 0

    def close(self):
        """Ferme SUMO"""
        try:
            traci.close()
        except:
            pass


# ==============================================================================
# ENTRAÎNEMENT
# ==============================================================================

class Trainer:
    """Gestionnaire d'entraînement"""

    def __init__(self, config):
        self.config = config
        self.agent = ImprovedQLearningAgent(config)

        # Créer dossiers
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        os.makedirs(config.PLOTS_DIR, exist_ok=True)

    def run_episode(self, episode_num, show_gui=False):
        """Exécute un épisode d'entraînement"""

        env = SUMOEnvironment(self.config, use_gui=show_gui)
        env.start()

        step = 0
        next_phase = 0
        state = self.agent.get_state(env.current_phase)

        while step < self.config.TOTAL_STEPS:
            # Step simulation
            env.step()

            # Décision
            if env.can_change_phase():
                # Choisir action
                action = self.agent.choose_action(state)
                next_phase = env.change_phase(action)

                # Calculer récompense
                reward, waiting, throughput = env.calculate_reward(phase_changed=True)

                # Nouvel état
                next_state = self.agent.get_state(next_phase)

                # Apprendre
                self.agent.learn(state, action, reward, next_state)

                # Update stats
                self.agent.update_episode_stats(reward, waiting, throughput)

                state = next_state

            # Appliquer phase verte après jaune
            if env.is_yellow and env.time_on_phase >= self.config.YELLOW_TIME:
                env.apply_phase(next_phase)
                state = self.agent.get_state(next_phase)

            step += 1

        env.close()

        # Finaliser épisode
        self.agent.decay_epsilon()
        self.agent.finish_episode()

        # Afficher progression
        if episode_num % 10 == 0 or episode_num == 1:
            avg_reward = np.mean(self.agent.episode_rewards[-10:])
            print(f"Episode {episode_num}/{self.config.NUM_EPISODES} | "
                  f"Récompense: {self.agent.episode_rewards[-1]:.2f} | "
                  f"Moy(10): {avg_reward:.2f} | "
                  f"ε: {self.agent.epsilon:.3f}")

    def train(self):
        """Lance l'entraînement complet"""

        print("\n" + "=" * 70)
        print("🚀 ENTRAÎNEMENT Q-LEARNING - 17 PHASES COMPATIBLES")
        print("=" * 70)
        print(f"Episodes: {self.config.NUM_EPISODES}")
        print(f"Actions: {self.config.NUM_ACTIONS}")
        print(f"États: {self.agent.num_states}")
        print("=" * 70 + "\n")

        for episode in range(1, self.config.NUM_EPISODES + 1):
            # GUI pour dernier épisode
            show_gui = (episode == self.config.NUM_EPISODES)
            self.run_episode(episode, show_gui=show_gui)

        print("\n" + "=" * 70)
        print("✅ ENTRAÎNEMENT TERMINÉ!")
        print("=" * 70)

        # Sauvegarder
        self.agent.save_model(f'{self.config.MODELS_DIR}/qlearning_model.json')

        # Statistiques finales
        self.print_statistics()

        # Visualisations
        self.plot_results()

    def print_statistics(self):
        """Affiche les statistiques finales"""

        print(f"\n📊 STATISTIQUES FINALES:")
        print("-" * 70)
        print(f"Meilleure récompense: {max(self.agent.episode_rewards):.2f}")
        print(f"Récompense moyenne: {np.mean(self.agent.episode_rewards):.2f}")
        print(f"Récompense finale (moy 10 derniers): {np.mean(self.agent.episode_rewards[-10:]):.2f}")
        print(f"Epsilon final: {self.agent.epsilon:.4f}")
        print(f"Temps d'attente moyen: {np.mean(self.agent.episode_waiting_times):.2f}s")
        print(f"Débit moyen: {np.mean(self.agent.episode_throughputs):.0f} véh/épisode")

    def plot_results(self):
        """Génère les visualisations"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        episodes = range(1, len(self.agent.episode_rewards) + 1)

        # Plot 1: Récompenses
        axes[0, 0].plot(episodes, self.agent.episode_rewards, alpha=0.3, color='blue')
        # Moving average
        window = 10
        if len(episodes) >= window:
            ma = pd.Series(self.agent.episode_rewards).rolling(window).mean()
            axes[0, 0].plot(episodes, ma, color='red', linewidth=2, label=f'MA({window})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Récompense')
        axes[0, 0].set_title('Récompenses par Episode', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Epsilon
        axes[0, 1].plot(episodes, self.agent.epsilon_history, color='green', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Epsilon')
        axes[0, 1].set_title('Décroissance Epsilon', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Temps d'attente
        axes[1, 0].plot(episodes, self.agent.episode_waiting_times, color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Temps d\'attente total (s)')
        axes[1, 0].set_title('Temps d\'Attente par Episode', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Débit
        axes[1, 1].plot(episodes, self.agent.episode_throughputs, color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Véhicules passés')
        axes[1, 1].set_title('Débit par Episode', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Apprentissage Q-Learning - Métriques de Performance',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOTS_DIR}/qlearning_training.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n✅ Plots sauvegardés: {self.config.PLOTS_DIR}/qlearning_training.png")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║    🧠 Q-LEARNING AMÉLIORÉ - FEUX DE CIRCULATION 🚦         ║
    ║                                                              ║
    ║    • 17 Phases Intelligentes avec Mouvements Compatibles    ║
    ║    • État Discréti (Pression × Phase)                       ║
    ║    • Récompense Multi-Objectifs                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Créer configuration
    config = Config()

    # Créer fichier de configuration SUMO
    sumocfg_content = """<configuration>
    <input>
        <net-file value="./SUMO intersection/intersection.net.xml"/>
        <route-files value="./SUMO intersection/intersection.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""

    with open("intersection.sumocfg", "w") as f:
        f.write(sumocfg_content)

    # Créer trainer et lancer entraînement
    trainer = Trainer(config)
    trainer.train()

    print("\n" + "=" * 70)
    print("✅ TERMINÉ!")
    print("=" * 70)
    print(f"📁 Modèle: {config.MODELS_DIR}/qlearning_model.json")
    print(f"📊 Plots: {config.PLOTS_DIR}/qlearning_training.png")