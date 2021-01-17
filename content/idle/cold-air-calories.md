---
date: "2021-01-17"
title: "Cold Air Calories"
slug: "cold-air-calories"
draft: false
hasMath: true
---

When you breathe in cold air, your body warms up that air. Simultaneously, your body temperature will lower slightly but then eventually come back to its _basal_ temperature of $\sim$98.6$^\circ$ F (37 $^\circ$ C). Your body must expend energy to raise the temperature back up. I was thinking about this and wondered:

_How many calories do we burn just by raising the temperature of cold air that we breathe in?_

Calorimetry can answer this for us pretty quickly (I think. Nobody's checked this work, and I haven't done physics in a long time!). The energy required to change the temperature of some substance is governed by the following equation

$$\Delta E = m c \Delta T$$

where $\Delta E$ is the change in energy, $m$ is the mass of the substance, $c$ is the _specific heat_, and $\Delta T$ is the change in temperature. The specific heat is a property of the substance which you can kind of think of like the slope, if you consider the above equation to be like a line. The specific heat tells us how much energy is required to change the temperature of the substance, and it's an empirically derived number that depends on the substance.

I would like to plug some numbers into the above equation, but I don't necessarily know how much air mass (_m_) is in each breath. I can find out the _volume_ ($V$) of air in a breath, though, and mass and volume are related by density ($\rho$):

$$m = \rho V$$

Plugging this into the first equation give us

$$\Delta E = \rho V c \Delta T $$

I found the following numbers online for the above variables:

- $\rho = 1.225 \frac{kg}{m^3}$
- $V = 0.5 L = 5 \times 10^{-4} m^3 $
- $c = .7171 \frac{kJ}{kg \cdot K}$

_Note: we use the specific heat of air assuming constant volume which seemed like a reasonable approximation given that your lungs expand? Honestly, I know very little about the body._

Plugging everything in and checking our units to make sure everything's correct, we get:

$$\Delta E = 
    \big(\frac{1.225 \ kg}{1 \ m^{3}} \big) 
    \big( \frac{5 \times 10^{-4} \ m^3 }{1}\big)
    \big(\frac{0.7171 \ kJ}{kg \cdot K})
    \Delta T
$$
$$\Delta E = 4.4 \times 10^{-4} \frac{kJ}{K} \Delta T$$

Lastly, let's convert $kJ$, our energy units, to $kcal$, aka American _calories_:

$$\Delta E = \big( 4.4 \times 10^{-4} \frac{kJ}{K} \big) \big(\frac{1  \ kcal}{4.184 \ kJ} \big)  \Delta T$$
$$ \Delta E = 1.05 \times 10^{-3} \frac{kcal}{K} \Delta T $$

Huzzah! We now have an equation that tells us how many calories are burned when we breathe in air that is colder than our body. Let's finally get to a single number and calculate how many calories are burned when we breathe in freezing air at 32$^\circ$F (0$^\circ$C). Our equation is in terms of Kelvin, so we can either plug in Kelvin or Celsius temperatures due to the fact that Celsius is just Kelvin offset by a constant factor. Your body's basal temperature is around 37$^\circ$C which gives us:

$$ \Delta E = 1.05 \times 10^{-3} \frac{kcal}{K} (37 \ C - 0 \ C) $$
$$ \Delta E = .039 \ kcal $$

And there you have it -- your body burns around $1/25$ of a calorie warming up the frozen air in each breath that you take.

The internet tells me that we take around 960 breaths per hour which corresponds to 37 calories of freezing air. Doesn't look like you can quit exercising anytime soon.