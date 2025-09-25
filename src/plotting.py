#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                     SCRIPT: plotting.py
#
#
#                DESCRIPTION: Real-time plotting utilities for RDK tasks.
#
#
#                       RULE: DAYW
#
#
#
#                    CREATOR: Sharif Saleki
#                       TIME: 09-23-2025-7810598105114117
#                      SPACE: Stanford Univeristy, Stanford, CA
#
# =================================================================================================== #
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from psychopy.tools.monitorunittools import pix2deg


@dataclass
class PlotConfig:
    last_n_trials: int = 20
    figsize: tuple = (24, 24)
    context: str = "poster"
    font_scale: float = 0.6


class TrialPlotter:
    """Encapsulates the real-time plotting of trial history.

    Usage:
        plotter = TrialPlotter(monitor=mon, subject=sub_id, session=ses_id, run_date=today)
        plotter.update(trial_history)
    """

    def __init__(
        self,
        monitor: Any,
        subject: str,
        session: int | str,
        run_date: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.monitor = monitor
        self.subject = subject
        self.session = session
        self.run_date = run_date or date.today().strftime("%Y-%m-%d")

        # config
        base = PlotConfig()
        if isinstance(config, dict):
            self.cfg = PlotConfig(
                last_n_trials=int(config.get("last_n_trials", base.last_n_trials)),
                figsize=tuple(config.get("figsize", base.figsize)),
                context=str(config.get("context", base.context)),
                font_scale=float(config.get("font_scale", base.font_scale)),
            )
        else:
            self.cfg = base

        # interactive mode and figure placeholders
        plt.ion()
        self._fig: Optional[plt.Figure] = None
        self._axs: Optional[np.ndarray] = None

    def figure(self) -> Optional[plt.Figure]:
        """Return the current matplotlib Figure if it exists."""
        return self._fig

    def _ensure_figure(self) -> None:
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._fig, self._axs = plt.subplots(3, 3, figsize=self.cfg.figsize)
            try:
                self._fig.canvas.manager.set_window_title("Trial Summary")
            except Exception:
                pass

    @staticmethod
    def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    def update(self, history: List[Dict[str, Any]]) -> None:
        """Update plots from a list of trial dictionaries.

        Expected trial fields include: 'trial_index', 'correct', 'direction',
        'speed', 'coherence', 'saccade_x', 'saccade_y', 'stim_x', 'stim_y',
        'saccade_rt'. Extra fields are ignored.
        """
        if not history:
            return

        # style
        sns.set_style("ticks")
        sns.despine()
        sns.set_context(self.cfg.context, font_scale=self.cfg.font_scale)

        # dataframe and derived fields
        df = pd.DataFrame(history)
        df["direction_name"] = self._col(df, "direction").map({90: "Up", 270: "Down"})
        df["speed_dva_per_sec"] = self._col(df, "speed").apply(lambda x: x * 100)
        df["saccade_y_dva"] = self._col(df, "saccade_y").apply(lambda y: pix2deg(y, monitor=self.monitor))
        df["saccade_x_dva"] = self._col(df, "saccade_x").apply(lambda x: pix2deg(x, monitor=self.monitor))
        df["stim_x_dva"] = self._col(df, "stim_x").apply(lambda x: pix2deg(x, monitor=self.monitor))
        df["stim_y_dva"] = self._col(df, "stim_y").apply(lambda y: pix2deg(y, monitor=self.monitor))

        self._ensure_figure()
        assert self._axs is not None and self._fig is not None

        axs = self._axs
        for row in axs:
            for ax in row:
                ax.clear()

        # title
        self._fig.suptitle(
            f"Subject: {self.subject}, Session: {self.session}, Date: {self.run_date}", fontsize=16
        )

        # panel [0,0]: correct/incorrect count of last N trials
        last_n = self.cfg.last_n_trials
        sns.countplot(
            x="trial_index",
            hue="correct",
            data=df.tail(last_n),
            palette={0: "red", 1: "green"},
            ax=axs[0, 0],
        )
        axs[0, 0].set_title(f"Correct/Incorrect Trials (Last {last_n} Trials)")
        axs[0, 0].set_xlabel("Trial")
        axs[0, 0].set_ylabel("Success")
        try:
            axs[0, 0].legend(title="Correct", loc="upper right", labels=["No", "Yes"])
        except Exception:
            pass

        # panel [0,1]: cumulative percent correct
        df["percent_correct"] = df["correct"].cumsum() / df["correct"].count() * 100
        sns.lineplot(x="trial_index", y="percent_correct", data=df, ax=axs[0, 1])
        axs[0, 1].set_title(
            f"Total correct {df['correct'].sum()} / {df['correct'].count()} trials | "
            f"Total reward {df['correct'].sum() * 0.25:.2f} ml"
        )
        axs[0, 1].set_xlabel("Trial")
        axs[0, 1].set_ylabel("Percent correct")
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].axhline(50, ls="--", color="black")
        axs[0, 1].axhline(100, ls="--", color="black")

        # panel [0,2]: saccade Y by coherence and speed (direction signed)
        if "saccade_y" in df.columns and "direction" in df.columns:
            valid = (
                df["saccade_y"].notna() & (df["saccade_y"] != -1) & df["direction"].isin([90, 270])
            )
        else:
            valid = pd.Series(False, index=df.index)
        if valid.sum() == 0:
            axs[0, 2].text(0.5, 0.5, "No saccade data yet", ha="center", va="center")
        else:
            tmp = df[valid].copy()
            tmp["signed_y"] = np.where(
                tmp["direction"] == 90, tmp["saccade_y_dva"], -tmp["saccade_y_dva"]
            )
            sns.barplot(
                data=tmp,
                x="coherence",
                y="signed_y",
                hue="speed_dva_per_sec",
                estimator=np.mean,
                errorbar="se",
                palette="muted",
                ax=axs[0, 2],
            )
            axs[0, 2].axhline(0, ls="--", color="gray", zorder=0)
            axs[0, 2].set_xlabel("Coherence")
            axs[0, 2].set_ylabel("Saccade Y (≈ Up - Down), dva")
            axs[0, 2].set_title("Saccade offset by Coherence and Speed")

        # panel [1,0]: RT distribution by direction
        rt_df = df[df.get("saccade_rt", 0) > 0]
        if rt_df.shape[0] == 0:
            axs[1, 0].text(0.5, 0.5, "No valid reaction times", ha="center", va="center")
            axs[1, 0].set_title("Saccade Latency Distribution")
            axs[1, 0].set_xlabel("Saccade Latency (ms)")
            axs[1, 0].set_ylabel("Count")
        else:
            try:
                sns.histplot(
                    data=rt_df, x="saccade_rt", hue="direction_name", bins=20, kde=True, ax=axs[1, 0]
                )
            except ValueError:
                sns.histplot(
                    data=rt_df, x="saccade_rt", hue="direction_name", bins=20, kde=False, ax=axs[1, 0]
                )
        axs[1, 0].set_title("Saccade RT Distribution")
        axs[1, 0].set_xlabel("RT (ms)")
        axs[1, 0].set_ylabel("Count")
        handles, labels = axs[1, 0].get_legend_handles_labels()
        if labels:
            axs[1, 0].legend(title="Direction", loc="upper right")

        # panel [1,1]: saccade landing scatter (correct, 100% coh)
        land_df = df[(df.get("correct", 0) == 1) & (df.get("coherence", 0) == 1)].copy()
        if "direction_name" not in land_df.columns and "direction" in land_df.columns:
            land_df["direction_name"] = land_df["direction"].map({90: "Up", 270: "Down"})

        if land_df.shape[0] == 0:
            axs[1, 1].text(0.5, 0.5, "No valid saccades", ha="center", va="center")
            axs[1, 1].set_title("Saccade Landing Positions")
            axs[1, 1].set_xlabel("X Position (dva)")
            axs[1, 1].set_ylabel("Y Position (dva)")
            axs[1, 1].axvline(0, ls="--", color="gray")
            axs[1, 1].axhline(0, ls="--", color="gray")
            axs[1, 1].set_aspect("equal", "box")
        else:
            has_hue = "direction_name" in land_df.columns and land_df["direction_name"].notna().any()
            if has_hue:
                sns.scatterplot(
                    x="saccade_x_dva",
                    y="saccade_y_dva",
                    hue="direction_name",
                    data=land_df,
                    ax=axs[1, 1],
                    palette="Set1",
                    s=100,
                )
            else:
                sns.scatterplot(
                    x="saccade_x_dva", y="saccade_y_dva", data=land_df, ax=axs[1, 1], color="C0", s=100
                )

            axs[1, 1].scatter(0, 0, s=200, c="black", marker="+", zorder=5)
            axs[1, 1].scatter(
                x=land_df["stim_x_dva"], y=land_df["stim_y_dva"], s=500, c="gray", marker="o", zorder=4
            )
            axs[1, 1].set_title("Saccade Landing Positions")
            axs[1, 1].set_xlabel("X Position (dva)")
            axs[1, 1].set_ylabel("Y Position (dva)")
            axs[1, 1].axvline(0, ls="--", color="gray")
            axs[1, 1].axhline(0, ls="--", color="gray")
            axs[1, 1].set_aspect("equal", "box")

            handles, labels = axs[1, 1].get_legend_handles_labels()
            if has_hue and len(labels) > 0:
                axs[1, 1].legend(title="Direction", loc="upper right")
            else:
                if axs[1, 1].legend_ is not None:
                    axs[1, 1].legend_.remove()

        # panel [1,2]: MIB distribution (angle)
        if land_df.shape[0] == 0:
            axs[1, 2].text(0.5, 0.5, "No valid saccades", ha="center", va="center")
            axs[1, 2].set_title("MIB Distribution")
            axs[1, 2].set_xlabel("Saccade Angle (degrees)")
            axs[1, 2].set_ylabel("Count")
        else:
            land_df["saccade_angle"] = (
                np.degrees(np.arctan2(-land_df["saccade_y_dva"], land_df["saccade_x_dva"])) + 360
            ) % 360
            sns.histplot(
                data=land_df, x="saccade_angle", hue="direction_name", bins=20, kde=False, ax=axs[1, 2], palette="Set1"
            )
            axs[1, 2].set_title("MIB Distribution")
            axs[1, 2].set_xlabel("Saccade Angle (degrees)")
            axs[1, 2].set_ylabel("Count")
            handles, labels = axs[1, 2].get_legend_handles_labels()
            if labels:
                axs[1, 2].legend(title="Direction", loc="upper right")
            else:
                if axs[1, 2].legend_ is not None:
                    axs[1, 2].legend_.remove()

        # panel [2,0]: RT by speed
        rt_df = df[df.get("saccade_rt", 0) > 0]
        if rt_df.shape[0] == 0:
            axs[2, 0].text(0.5, 0.5, "No valid reaction times", ha="center", va="center")
            axs[2, 0].set_title("Saccade Latency Distribution")
            axs[2, 0].set_xlabel("Saccade Latency (ms)")
            axs[2, 0].set_ylabel("Count")
        else:
            try:
                sns.histplot(
                    data=rt_df,
                    x="saccade_rt",
                    hue="speed_dva_per_sec",
                    bins=20,
                    kde=True,
                    ax=axs[2, 0],
                    palette="RdYlBu",
                )
            except ValueError:
                sns.histplot(
                    data=rt_df,
                    x="saccade_rt",
                    hue="speed_dva_per_sec",
                    bins=20,
                    kde=False,
                    ax=axs[2, 0],
                    palette="RdYlBu",
                )
        axs[2, 0].set_title("Saccade RT Distribution")
        axs[2, 0].set_xlabel("RT (ms)")
        axs[2, 0].set_ylabel("Count")
        handles, labels = axs[2, 0].get_legend_handles_labels()
        if labels:
            axs[2, 0].legend(title="Speed", loc="upper right")

        # panel [2,1]: angle vs latency (correct, 100% coh)
        angle_df = df[
            (df.get("correct", 0) == 1)
            & (df.get("coherence", 0) == 1)
            & (df.get("saccade_rt", 0) > 0)
        ].copy()
        angle_df["saccade_angle"] = (
            np.degrees(np.arctan2(-angle_df["saccade_y_dva"], angle_df["saccade_x_dva"])) + 360
        ) % 360
        if angle_df.shape[0] == 0:
            axs[2, 1].text(0.5, 0.5, "No valid saccades", ha="center", va="center")
            axs[2, 1].set_title("Saccade Angle vs. Latency")
            axs[2, 1].set_xlabel("Saccade Latency (ms)")
            axs[2, 1].set_ylabel("Saccade Angle (degrees)")
        else:
            sns.scatterplot(
                x="saccade_rt",
                y="saccade_angle",
                hue="direction_name",
                data=angle_df,
                ax=axs[2, 1],
                palette="Set1",
                s=100,
            )
            axs[2, 1].set_title("Saccade Angle vs. Latency")
            axs[2, 1].set_xlabel("Saccade Latency (ms)")
            axs[2, 1].set_ylabel("Saccade Angle (degrees)")
            axs[2, 1].set_ylim(0, 360)
            handles, labels = axs[2, 1].get_legend_handles_labels()
            if len(labels) > 0:
                axs[2, 1].legend(title="Direction", loc="upper right")
            else:
                if axs[2, 1].legend_ is not None:
                    axs[2, 1].legend_.remove()

        # panel [2,2]: MIB by coherence with optional Weibull fit
        mib_df = df[
            (df.get("correct", 0) == 1)
            & (df.get("saccade_rt", 0) > 0)
            & (df.get("saccade_y").notna() if "saccade_y" in df.columns else False)
        ].copy()
        if "direction" in mib_df.columns:
            mib_df["saccade_y_signed"] = np.where(
                mib_df["direction"] == 90, mib_df["saccade_y_dva"], -mib_df["saccade_y_dva"]
            )
        else:
            mib_df["saccade_y_signed"] = 0.0
        psycho_df = (
            mib_df.groupby("coherence").agg(
                n_trials=("saccade_y_signed", "count"),
                mean_mib=("saccade_y_signed", "mean"),
                sem_mib=("saccade_y_signed", "sem"),
            ).reset_index()
            if not mib_df.empty and "coherence" in mib_df.columns
            else pd.DataFrame(columns=["coherence", "n_trials", "mean_mib", "sem_mib"])
        )

        if psycho_df.shape[0] < 2:
            axs[2, 2].text(0.5, 0.5, "Not enough data", ha="center", va="center")
            axs[2, 2].set_title("MIB by Coherence")
            axs[2, 2].set_xlabel("Coherence")
            axs[2, 2].set_ylabel("MIB (saccade Y, dva)")
        else:
            sns.pointplot(
                data=psycho_df, x="coherence", y="mean_mib", yerr=psycho_df["sem_mib"], color="C0", ax=axs[2, 2]
            )
            axs[2, 2].axhline(0, ls="--", color="gray", zorder=0)
            axs[2, 2].set_title("MIB by Coherence (correct trials)")
            axs[2, 2].set_xlabel("Coherence")
            axs[2, 2].set_ylabel("MIB (saccade Y, dva)")

            if psycho_df["coherence"].nunique() >= 3:
                from scipy.optimize import curve_fit  # lazy import

                def weibull(x, alpha, beta, gamma, delta):
                    return gamma + (1 - gamma - delta) * (1 - np.exp(-(x / alpha) ** beta))

                try:
                    popt, _ = curve_fit(
                        weibull,
                        psycho_df["coherence"],
                        psycho_df["mean_mib"],
                        bounds=([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
                        maxfev=10000,
                    )
                    x_fit = np.linspace(
                        psycho_df["coherence"].min(), psycho_df["coherence"].max(), 100
                    )
                    y_fit = weibull(x_fit, *popt)
                    axs[2, 2].plot(x_fit, y_fit, "r--", label="Weibull fit")
                    axs[2, 2].legend()
                    fit_text = (
                        f"Weibull fit:\nα={popt[0]:.2f}, β={popt[1]:.2f}\nγ={popt[2]:.2f}, δ={popt[3]:.2f}"
                    )
                    axs[2, 2].text(
                        0.05,
                        0.95,
                        fit_text,
                        transform=axs[2, 2].transAxes,
                        verticalalignment="top",
                        fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
                    )
                except Exception as e:
                    print(f"Warning: failed to fit Weibull function: {e}")
            else:
                axs[2, 2].text(0.5, 0.5, "Not enough coherence levels to fit", ha="center", va="center")
                axs[2, 2].set_title("MIB by Coherence")
                axs[2, 2].set_xlabel("Coherence")
                axs[2, 2].set_ylabel("MIB (saccade Y, dva)")
                axs[2, 2].set_ylim(
                    psycho_df["mean_mib"].min() - 1, psycho_df["mean_mib"].max() + 1
                )

        # layout and draw
        plt.tight_layout()
        try:
            self._fig.canvas.draw_idle()
        except Exception:
            self._fig.canvas.draw()
        plt.pause(0.001)


__all__ = ["TrialPlotter", "PlotConfig"]
