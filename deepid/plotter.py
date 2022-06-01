#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package plotter
Plotter for searchingfornues TrkAna TTree

This module plots variables from the TTree produced by the
TrkAna module of the Mu2e offline (https://mu2ewiki.fnal.gov/wiki/TrkAna)

Example:
    my_plotter = plotter.Plotter(samples, weights)
    fig, ax = my_plotter.plot_variable("deent_mom",
                                       title="Reco. momentum [GeV]",
                                       cat_var="demcgen_gen",
                                       range=(95,110),
                                       bins=30)

Attributes:
    category_labels (dict): Description of event categories
    pdg_labels (dict): Labels for PDG codes
    category_colors (dict): Color scheme for event categories
    pdg_colors (dict): Colors scheme for PDG codes
"""

import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    """Main plotter class

    Args:
        samples (dict): Dictionary of pandas dataframes.
        weights (dict): Dictionary of global dataframes weights.
            One for each entry in the samples dict.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       weights (dict): Dictionary of global dataframes weights.
       pot (int): Number of protons-on-target.
    """

    genid_labels = {
        7: r"Decay-in-orbit",
        11: r"Radiative pion capture (external)",
        22: r"Radiative pion capture (internal)",
        38: r"Cosmic ray",
        41: r"Radiative muon capture (external)",
        42: r"Radiative muon capture (internal)",
        43: r"Conversion electron",
    }

    genid_colors = {
        7: "xkcd:cobalt",
        11: "xkcd:cerulean",
        22: "xkcd:sky blue",
        38: "xkcd:cyan",
        41: "xkcd:lime green",
        42: "xkcd:green",
        43: "xkcd:light red",
    }

    pdg_labels = {
        -211: r"$\pi^{+}$",
        -13: r"$\mu^{+}$",
        -11: r"$e^{+}$",
        11: r"$e^{-}$",
        13: r"$\mu^{-}$",
        22: r"$\gamma$",
        211: r"$\pi^{-}$",
        2112: r"$n$",
        2212: r"$p$"
    }

    pdg_colors = {
        -211: "#ff7f00",
        -13: "#1f78b4",
        -11: "#a6cee3",
        11: "#33a02c",
        13: "#b2df8a",
        22: "#cab2d6",
        211: "#fdbf6f",
        2112: "#137e6d",
        2212: "#e31a1c"
    }

    colors = {
        'demcgen_gen': genid_colors,
        'gen': genid_colors,
        'demcgen_pdg': pdg_colors,
        'pdg': pdg_colors
    }

    labels = {
        'demcgen_gen': genid_labels,
        'gen': genid_labels,
        'demcgen_pdg': pdg_labels,
        'pdg': pdg_labels
    }

    def __init__(self, samples, weights):
        self.samples = samples
        self.weights = weights

        # for s in self.samples:
        #     self.samples[s] = self.samples[s].assign(
        #         weight=[self.weights[s]] * len(self.samples[s]))


    def _categorize(self, df, cat_var, var_name):
        var_dict = OrderedDict()
        weight_dict = OrderedDict()
        grouped = df.groupby(cat_var)
        for cat in sorted(df[cat_var].unique(), reverse=True):
            var_dict[cat] = grouped.get_group(cat).eval(var_name)
            weight_dict[cat] = grouped.get_group(cat)['weight']

        return var_dict, weight_dict


    def plot_variable(self,
                      ax,
                      variable,
                      query="deent_mom>0",
                      title="",
                      cat_var="demcgen_gen",
                      plot_range=None,
                      x_range=[0, 0],
                      bins=0):
        """It plots the variable from the TTree, after applying an eventual query

        Args:
            variable (str): name of the variable.
            query (str): pandas query. Default is ``selected``.
            title (str, optional): title of the plot. Default is ``variable``.
            cat_var (str, optional): Categorization of the plot. Can be ``demcgen_pdg``
                or ``demcgen_gen``
            x_range: Range of the plot on the x-axis
            bins: Number of histogram bins

        Returns:
            Figure and subplots

        """
        if not title:
            title = variable

        if query:
            if x_range == [0, 0]:
                x_range = [
                    min(self.samples["mc"].query(query)[variable]),
                    max(self.samples["mc"].query(query)[variable])
                ]
            query += " & "
        else:
            if x_range == [0, 0]:
                x_range = [
                    min(self.samples["mc"][variable]),
                    max(self.samples["mc"][variable])
                ]

        # pandas bug https://github.com/pandas-dev/pandas/issues/16363
        if x_range[0] < 0 and x_range[1] > 0:
            query += "%s <= %g & -%s <= -%g" % (variable,
                                                x_range[1],
                                                variable,
                                                x_range[0])
        elif x_range[0] > 0 and x_range[1] < 0:
            query += "-%s >= -%g & %s >= %g" % (variable,
                                                x_range[1],
                                                variable,
                                                x_range[0])
        elif x_range[0] < 0 and x_range[1] < 0:
            query += "-%s >= -%g & -%s <= -%g" % (variable,
                                                  x_range[1],
                                                  variable,
                                                  x_range[0])
        else:
            query += "%s <= %g & %s >= %g" % (variable,
                                              x_range[1],
                                              variable,
                                              x_range[0])

        if plot_range:
            query_outside = query + f"& ({variable} <= {plot_range[0]} | {variable} >= {plot_range[1]})"
            query += f"& {variable} >= {plot_range[0]} & {variable} <= {plot_range[1]}"
            mc_outside = self.samples["mc"].query(query_outside)
            var_dict_outside, weight_dict_outside = self._categorize(mc_outside, cat_var, variable)
                
        mc = self.samples["mc"].query(query)
        
        if not bins:
            bins = int(math.sqrt(len(mc)))

        var_dict, weight_dict = self._categorize(mc, cat_var, variable)

        cat_labels = [
            "%s: %g" % (self.labels[cat_var][c], sum(weight_dict[c]))
            for c in sorted(mc[cat_var].unique(), reverse=True)
        ]

        cat_colors = [self.colors[cat_var].get(c) for c in sorted(mc[cat_var].unique(), reverse=True)]
        if not var_dict.values():
            raise ValueError("No entries selected")
            
        if plot_range:
            cat_outside_colors = [self.colors[cat_var].get(c) for c in sorted(mc_outside[cat_var].unique(), reverse=True)]
            if var_dict_outside.values():
                ax.hist(var_dict_outside.values(),
                    stacked=True,
                    edgecolor=None,
                    linewidth=0,
                    # label=cat_labels,
                    color=cat_outside_colors,
                    alpha=0.5,
                    weights=list(weight_dict_outside.values()),
                    range=x_range,
                    bins=bins)
                
            total_array = np.concatenate(list(var_dict_outside.values()))
            total_weight = np.concatenate(list(weight_dict_outside.values()))

            n_tot, tot_bins, patches = ax.hist(total_array,
                                               weights=total_weight,
                                               histtype="step",
                                               edgecolor="black",
                                               range=x_range,
                                               alpha=0.5,
                                               bins=bins)   
            
            bincenters = 0.5 * (tot_bins[1:] + tot_bins[:-1])
            mc_err = np.sqrt(n_tot * self.weights["mc"] * self.weights["mc"])

            # ax.errorbar(bincenters,
            #        n_tot,
            #        linewidth=0,
            #        edgecolor=None,
            #        width=0,
            #        alpha=0.5,
            #        yerr=mc_err)
            ax.axvline(plot_range[0], ls="--", c='k')
            ax.axvline(plot_range[1], ls="--", c='k')


        ax.hist(var_dict.values(),
                stacked=True,
                edgecolor=None,
                linewidth=0,
                label=cat_labels,
                color=cat_colors,
                weights=list(weight_dict.values()),
                range=x_range,
                bins=bins)

        
        total_array = np.concatenate(list(var_dict.values()))
        total_weight = np.concatenate(list(weight_dict.values()))
        
        n_tot, tot_bins, patches = ax.hist(total_array,
                                           weights=total_weight,
                                           histtype="step",
                                           edgecolor="black",
                                           range=x_range,
                                           bins=bins)

        bincenters = 0.5 * (tot_bins[1:] + tot_bins[:-1])
        mc_err = np.sqrt(n_tot * self.weights["mc"] * self.weights["mc"])

        # ax.bar(bincenters,
        #        n_tot,
        #        linewidth=0,
        #        edgecolor=None,
        #        width=0,
        #        yerr=mc_err)

        if plot_range:
            leg = ax.legend(title=f'Mu2e simulation\n{plot_range[0]} MeV < p < {plot_range[1]} MeV')
        else:
            leg = ax.legend(title=r'Mu2e simulation')
        plt.setp(leg.get_title(), fontweight='bold')
        ax.set_ylim(0., max(n_tot)*1.5)
        ax.set_xlim(x_range[0], x_range[1])

        unit = title[title.find("[") + 1:title.find("]")
                     ] if "[" and "]" in title else ""

        x_range_size = x_range[1] - x_range[0]

        if isinstance(bins, Iterable):
            ax.set_ylabel("N. Entries")
        else:
            ax.set_ylabel("N. Entries / %g %s" % (x_range_size / bins, unit))

        ax.set_xlabel(title)
