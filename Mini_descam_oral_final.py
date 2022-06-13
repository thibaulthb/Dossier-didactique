#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:47:58 2022

@author: thibault
"""

## Importation des bibliotheques
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from numpy import pi
# format vectoriel par défaut des images
from IPython.display import set_matplotlib_formats

# Paramètres généraux de pyplot
set_matplotlib_formats('svg')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Définition des constantes principales
grav=9.81   #m/s²
Cp=1006     #J/K/kg
Cw=4180     #J/K/kg
Lv=2500e3   #J/kg
R=8.314     #J/K/mol
M=29e-3     #kg/mol
Mw=18e-3    #kg/mol
gamma=1.4   #Sans Unités
T0=273.15   #K
Te0=300     #K
p0=101300   #Pa
tmax=1800   #s

def T_std(z,Gamma):
    return Te0-Gamma*z

def T_par(z,z_amont,T_amont,Gamma):
    return T_amont*(p_std(z_amont,Gamma)/p_std(z,Gamma))**((1-gamma)/gamma)

def p_std(z,Gamma):
    return p0*(1-Gamma*z/Te0)**(M*grav/(Gamma*R))

def dp_std(z,Gamma):
    return -p0*(Gamma/Te0)*(M*grav/(Gamma*R))*(1-Gamma/Te0*z)**(M*grav/(Gamma*R)-1)

def theta(T,p):
    return T*(p0/p)**(R/(Cp*M))

def rho(z,T,r,p):
    return p*M/(R*T)*(1+r)/(1+M/Mw*r)

def rho_dry(T,p):
    return p*M/(R*T)

def psatw(T):
    return 611*np.exp(17.502*(T-T0)/(T-32.19))

def psati(T):
    return 611*np.exp(22.587*(T-T0)/(T+0.7))

def slwv(T):
    return 2.5e6-2.34e3*(T-T0)

def mix_rat(p_w,p):
    return Mw/M*p_w/p

# Fonction principale de calcul de l'évolution de la parcelle d'air
def Mini_descam(dt,T1,RH_ini,Gamma,mode,condrate,condmax):
    global alt_dyn,vit_dyn,T_dyn,pw_dyn,qc_dyn,r_dyn
    
    if mode >= 6:#Bridage de la condensation
        cond_rate=min(dt/condrate,1)
    else:
        cond_rate=1
    if mode > 0:
        T_dyn[0] = T1
        if mode > 1:
            pw_dyn[0]  = RH_ini*psatw(T_dyn[0])
            r_dyn[0]   = mix_rat(pw_dyn[0],p_std(alt_dyn[0],Gamma))
        
        nmax=int(np.floor(tmax/dt))+1
        # Calcul de l'évolution temporelle de la parcelle d'air
        for i in range (1,nmax):
            if mode == 0:
                vit_dyn[i] = vit_dyn[i-1] + (-1/(rho_dry(T_dyn[i-1],p_std(alt_dyn[i-1],Gamma)))*dp_std(alt_dyn[i-1],Gamma)-grav)*dt
            else:
                vit_dyn[i] = vit_dyn[i-1] + (-1/(rho(alt_dyn[i-1],T_dyn[i-1],r_dyn[i-1],p_std(alt_dyn[i-1],Gamma))+qc_dyn[i-1])*dp_std(alt_dyn[i-1],Gamma)-grav)*dt
            alt_dyn[i] = alt_dyn[i-1] + vit_dyn[i]*dt
            T_dyn[i]   = T_par(alt_dyn[i],alt_dyn[i-1],T_dyn[i-1],Gamma)
            pw_dyn[i]  = pw_dyn[i-1]
            qc_dyn[i]  = qc_dyn[i-1]
            r_dyn[i]   = mix_rat(pw_dyn[i],p_std(alt_dyn[i],Gamma))
            
            # Calcul de la condensation
            if mode > 2:# Prise en compte de la condensation
                cond=0
                if pw_dyn[i] > psatw(T_dyn[i]):
                    cond = cond_rate*(pw_dyn[i]-psatw(T_dyn[i]))*Mw/(R*T_dyn[i])
                else:
                    if mode > 4:# Prise en compte de l'évaporation
                        cond = -cond_rate*min(qc_dyn[i],-(pw_dyn[i]-psatw(T_dyn[i]))*Mw/(R*T_dyn[i]))
                if cond != 0:
                    if abs(cond)<condmax or mode < 7:#Mise à jour des grandeurs et dégagement de chaleur latente
                        T_store   = T_dyn[i]
                        if mode > 3:# Mise à jour du LWC
                            qc_dyn[i] = qc_dyn[i]+cond
                        T_dyn[i]  = T_dyn[i]+slwv(T_dyn[i])*cond/(Cp*rho(alt_dyn[i],T_dyn[i],r_dyn[i],p_std(alt_dyn[i],Gamma))+Cw*qc_dyn[i])
                        pw_dyn[i] = pw_dyn[i] - cond*(R*T_store)/Mw
                        r_dyn[i]  = mix_rat(pw_dyn[i],p_std(alt_dyn[i],Gamma))
                    else:
                        if mode == 7:#Subdivision du pas de temps
                            if i < 500 and i > 480:
                                f = open("condupdt_"+str(i)+".txt", 'w')
                            nconddt=min(int(np.floor(abs(cond)/condmax*2.5)),20)
                            T_var   = (T_dyn[i]-T_dyn[i-1])/nconddt
                            alt_var = (alt_dyn[i]-alt_dyn[i-1])/nconddt
                            T_dt    = T_dyn[i-1]
                            alt_dt  = alt_dyn[i-1]
                            pw_dt   = pw_dyn[i-1]
                            qc_dt   = qc_dyn[i-1]
                            r_dt    = r_dyn[i-1]
                            for idt in range (0,nconddt):
                                T_dt   = T_dt+T_var
                                T_store = T_dt
                                alt_dt = alt_dt+alt_var
                                cond=0
                                if pw_dt > psatw(T_dt):
                                    cond = cond_rate*(pw_dt-psatw(T_dt))*Mw/(R*T_dt)/nconddt
                                else:
                                    cond = -cond_rate*min(qc_dt,-(pw_dt-psatw(T_dt))*Mw/(R*T_dt))/nconddt
                                qc_dt = qc_dt+cond
                                T_dt  = T_dt+slwv(T_dt)*cond/(Cp*rho(alt_dt,T_dt,r_dt,p_std(alt_dt,Gamma))+Cw*qc_dt)
                                pw_dt = pw_dt - cond*(R*T_store)/Mw
                                r_dt  = mix_rat(pw_dt,p_std(alt_dt,Gamma))
                                if i < 500 and i > 480:
                                    np.savetxt(f,[idt,T_dt,T_store,alt_dt,cond,qc_dt,pw_dt,psatw(T_store),r_dt,nconddt],delimiter="\n",newline="  ")
                                    f.write("\n")
                            T_dyn[i]  = T_dt
                            pw_dyn[i] = pw_dt
                            qc_dyn[i] = qc_dt
                            r_dyn[i]  = mix_rat(pw_dyn[i],p_std(alt_dyn[i],Gamma))
                            if i < 500 and i > 480:
                                f.close()
                                
def get_mode(label):
    if label == "Création du profil":
        return 0
    elif label == "Parcelle d'air sec":
        return 1
    elif label == "Parcelle d'air humide":
        return 2
    elif label == "Condensation irréversible":
        return 3
    elif label == "Condensation sans perte":
       return 4
    elif label == "Condensation/évaporation":
        return 5
    elif label == "Condensation bridée":
        return 6
    elif label == "Condensation subdivisée":
        return 7

# Initialisation des valeurs à stocker

global alt_dyn,vit_dyn,T_dyn,pw_dyn,qc_dyn,r_dyn
dt=2
nmax=int(np.floor(tmax/dt))+1
alt_dyn = np.zeros(nmax)
vit_dyn = np.zeros(nmax)
T_dyn   = np.zeros(nmax)
pw_dyn  = np.zeros(nmax)
qc_dyn  = np.zeros(nmax)
r_dyn   = np.zeros(nmax)
temps   = np.zeros(nmax)
    
# Création de la fenêtre
#fig = plt.figure(constrained_layout=True, figsize=(14,10))
fig = plt.figure(figsize=(14,10))
plt.subplots_adjust(left   = 0.1,
                    bottom = 0.1,
                    right  = 0.9,
                    top    = 0.9,
                    wspace = 0.4,
                    hspace = 0.5)
gs = fig.add_gridspec(ncols=3, nrows=4)
ax_T = fig.add_subplot(gs[0,0])
ax_p = fig.add_subplot(gs[0,1])
ax_alt = fig.add_subplot(gs[1:3,0:2])
ax_RH = fig.add_subplot(gs[3,0])
ax_qc = fig.add_subplot(gs[3,1])
#plt.subplots_adjust(bottom=0.25,left=0.1,right=.7)         # dimensions du graphique

# Mise en place des contrôles
axcolor = 'white'
rax = plt.axes([.7, 0.65, 0.2, 0.25])
radio = RadioButtons(rax, ("Création du profil", "Parcelle d'air sec",
                           "Parcelle d'air humide","Condensation irréversible",
                           "Condensation sans perte","Condensation/évaporation",
                           "Condensation bridée","Condensation subdivisée"))
radio.set_active(0)
mode=get_mode(radio.value_selected)
rax = plt.axes([.7, 0.46, 0.14, 0.15])
radio2 = RadioButtons(rax, ("Température","Température potentielle"))
radio2.set_active(0)
temperature=radio2.value_selected

axT = plt.axes([0.7, 0.39, .2, 0.03], facecolor=axcolor)
sl_axT = Slider(axT, '$T_{ini}$(K)', 299, 304, valinit=303, valstep=0.1)  
T_ini=sl_axT.val                       #Slider pour modifier le temps
axRH = plt.axes([0.7, 0.34, .2, 0.03], facecolor=axcolor)
sl_axRH = Slider(axRH, "$RH_{ini}$", 0.2, 0.5, valinit=0.262, valstep=0.001) 
RH_ini=sl_axRH.val                        #Slider pour modifier la vitesse
axGamma = plt.axes([0.7, 0.29, .2, 0.03], facecolor=axcolor)
sl_axGamma = Slider(axGamma, r"$\Gamma_z$(K/km)", 5, 12, valinit=6.5, valstep=0.1) 
Gamma=sl_axGamma.val*1e-3                        #Slider pour modifier la vitesse
axdt = plt.axes([0.7, 0.24, .2, 0.03], facecolor=axcolor)
sl_axdt = Slider(axdt, "dt (s)", 0.5, 10, valinit=dt, valstep=0.1) 
pas_temps=sl_axdt.val                        #Slider pour modifier la vitesse
axcond = plt.axes([0.7, 0.17, .2, 0.03], facecolor=axcolor)
sl_axcond = Slider(axcond, "Temps de condensation/évaporation (s)", 0.5, 20, valinit=2, valstep=0.05) 
condrate=sl_axcond.val                        #Slider pour modifier la vitesse
sl_axcond.label.set_position([1,1.5])
axcondmax = plt.axes([0.7, 0.1, .2, 0.03], facecolor=axcolor)
sl_axcondmax = Slider(axcondmax, "Taux maximum de condensation (g/m$^3$)", -6, 1, valinit=1, valstep=0.1,valfmt='$10^{%0.1f}$') 
condmax=10**(sl_axcondmax.val)                       #Slider pour modifier la vitesse
sl_axcondmax.label.set_position([1,1.5])
                            
        
# Calcul initial
nmax=int(np.floor(tmax/pas_temps))
temps = np.arange(0,nmax*pas_temps,pas_temps)
Mini_descam(pas_temps, T_ini, RH_ini, Gamma,mode,condrate,condmax)

altitude=np.linspace(0,4000,101)

# Tracé du profil de température
if temperature=="Température":
    ax_T.plot(T_std(altitude,Gamma),altitude/1000,label="$T_{env}$")
    ax_T.plot(T_dyn[0:nmax],alt_dyn[0:nmax]/1000,'g-',label="$T_{parc}$")
else:
    ax_T.plot(theta(T_std(altitude,Gamma),p_std(altitude,Gamma)),altitude/1000,label=r"$\theta_{env}$")
    ax_T.plot(theta(T_dyn[0:nmax]*(1+1.608*r_dyn[0:nmax])/(1+r_dyn[0:nmax]),p_std(alt_dyn[0:nmax],Gamma)),alt_dyn[0:nmax]/1000,'g-',label=r"$\theta_{parc}$")
ax_T.legend(bbox_to_anchor=(.99, .99),loc='upper right', borderaxespad=0.)

ax_T.axis([275, 305, 0, 4])                     # limite des axes (xmin,xmax,ymin,ymax)
ax_T.set_xlabel("Température (K)")                     # titre de l'axe des abscisses
ax_T.set_ylabel("Altitude (km)")                               # titre de l'axe des ordonnees
ax_T.title.set_text("Profil de température") 

# Tracé du profil de pression
ax_p.plot(p_std(altitude,Gamma)/100,altitude/1000)

ax_p.axis([600, 1020, 0, 4])                     # limite des axes (xmin,xmax,ymin,ymax)
ax_p.set_xlabel("Pression (hPa)")                     # titre de l'axe des abscisses
ax_p.set_ylabel("Altitude (km)")                               # titre de l'axe des ordonnees
ax_p.title.set_text("Profil de pression") 

# Tracé de l'évolution dynamique de la parcelle
ax_alt.plot(temps[0:nmax]/60,alt_dyn[0:nmax]/1000,label="Altitude de la parcelle")
ax_alt.axis([0, 30, 0, 4])                     # limite des axes (xmin,xmax,ymin,ymax)
ax_alt.set_xlabel("Temps (min)")                     # titre de l'axe des abscisses
ax_alt.set_ylabel("Altitude (km)")                               # titre de l'axe des ordonnees
ax_alt.title.set_text("Évolution adiabatique d'une masse d'air humide") 
ax_alt.legend(bbox_to_anchor=(0.01, .99),loc='upper left', borderaxespad=0.)

ax_Tdyn=ax_alt.twinx()
ax_Tdyn.plot(temps[0:nmax]/60,T_dyn[0:nmax],'r-',label="Température de la parcelle")

ax_Tdyn.axis([0, 30, 270, 310])                     # limite des axes (xmin,xmax,ymin,ymax)
ax_Tdyn.set_xlabel("Temps (min)")                     # titre de l'axe des abscisses
ax_Tdyn.set_ylabel("Température")                               # titre de l'axe des ordonnees
ax_Tdyn.legend(bbox_to_anchor=(.99, .99),loc='upper right', borderaxespad=0.)

# Tracé de l'évolution de l'humidité relative

ax_RH.plot(temps[0:nmax]/60,pw_dyn[0:nmax]/psatw(T_dyn[0:nmax]))

ax_RH.axis([0, 30, 0, 1.25])                     # limite des axes (xmin,xmax,ymin,ymax)
ax_RH.set_xlabel("Temps (min)")                     # titre de l'axe des abscisses
ax_RH.set_ylabel("Humidité relative")                               # titre de l'axe des ordonnees

# Tracé de l'évolution du contenu en eau liquide

ax_qc.plot(temps[0:nmax]/60,qc_dyn[0:nmax]*1000)

ax_qc.axis([0, 30, 0, 3])                     # limite des axes (xmin,xmax,ymin,ymax)
ax_qc.set_xlabel("Temps (min)")                     # titre de l'axe des abscisses
ax_qc.set_ylabel("LWC (g/L)")                               # titre de l'axe des ordonnees

# Mise à jour des tracés par changement de slider ou de mode de simulation
def update(val):
    global alt_dyn,vit_dyn,T_dyn,pw_dyn,qc_dyn,r_dyn
    T_ini     = sl_axT.val          #Slider pour modifier la température initiale
    RH_ini    = sl_axRH.val         #Slider pour modifier l'humidité initiale'
    Gamma     = sl_axGamma.val*1e-3 #Slider pour modifier le taux de refroidissement de l'atmosphère
    pas_temps = sl_axdt.val         #Slider pour modifier le pas de temps
    condrate  = sl_axcond.val                        #Slider pour modifier la vitesse
    condmax   = 10**(sl_axcondmax.val)                        #Slider pour modifier la vitesse
    
    nmax=int(np.floor(tmax/pas_temps))+1
    alt_dyn = np.zeros(nmax)
    vit_dyn = np.zeros(nmax)
    T_dyn   = np.zeros(nmax)
    pw_dyn  = np.zeros(nmax)
    qc_dyn  = np.zeros(nmax)
    r_dyn   = np.zeros(nmax)
    temps   = np.zeros(nmax)
    
    temps = np.arange(0,nmax*pas_temps,pas_temps)
    Mini_descam(pas_temps, T_ini, RH_ini, Gamma, get_mode(radio.value_selected),condrate,condmax)

    if radio2.value_selected == "Température":
        ax_T.lines[0].set_xdata(T_std(altitude,Gamma))
        ax_T.lines[0].set_label(r"$T_{env}$")
        ax_T.lines[1].set_data(T_dyn[0:nmax],alt_dyn[0:nmax]/1000)
        ax_T.lines[1].set_label(r"$T_{parc}$")
        ax_T.set_xlim([275, 305])
    else:
        ax_T.lines[0].set_xdata(theta(T_std(altitude,Gamma),p_std(altitude,Gamma)))
        ax_T.lines[0].set_label(r"$\theta_{env}$")
        ax_T.lines[1].set_data(theta(T_dyn[0:nmax]*(1+1.608*r_dyn[0:nmax])/(1+r_dyn[0:nmax]),p_std(alt_dyn[0:nmax],Gamma)),alt_dyn[0:nmax]/1000)
        ax_T.lines[1].set_label(r"$\theta_{parc}$")
        ax_T.set_xlim([295, 325])
    ax_T.legend(bbox_to_anchor=(.99, .99),loc='upper right', borderaxespad=0.)

    
    ax_p.lines[0].set_xdata(p_std(altitude,Gamma)/100)
    
    ax_alt.lines[0].set_data(temps[0:nmax]/60,alt_dyn[0:nmax]/1000)
    ax_Tdyn.lines[0].set_data(temps[0:nmax]/60,T_dyn[0:nmax])
    
    ax_RH.lines[0].set_data(temps[0:nmax]/60,pw_dyn[0:nmax]/psatw(T_dyn[0:nmax]))
    
    ax_qc.lines[0].set_data(temps[0:nmax]/60,qc_dyn[0:nmax]*1000)
    
    plt.draw()
    
sl_axT.on_changed(update)
sl_axRH.on_changed(update)
sl_axGamma.on_changed(update)
sl_axdt.on_changed(update)
sl_axcond.on_changed(update)
sl_axcondmax.on_changed(update)

radio.on_clicked(update)
radio2.on_clicked(update)

# ## Definition d'un bouton reset
# axcolor = 'lightgoldenrodyellow'                            # couleur des barres
# resetax = plt.axes([.82, 0.66, 0.06, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
# def reset(event):
#     sl_axT.reset()
#     sl_axRH.reset()
#     sl_axGamma.reset()
#     sl_axdt.reset()
    
# button.on_clicked(reset)

# Affichage de la figure
plt.show()