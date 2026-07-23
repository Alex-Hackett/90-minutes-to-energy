# 90-minutes-to-energy
A very quick random forest binary classifier for a 90 minute online challenge, with the goal to use a year of minute cadence data to predict buy/sell on the energy regulation market

This was a time limited challenge where one was supplied with a minute-by-minute cadence dataset of energy regulation reporting. In essence, when one sells on this market, and the regulation energy required is positive at the end of the minute, the national energy regulator penalizes the fund / trader in question for doing so, with a fine, and on the other hand, when holding the asset, if there is regulation energy required, the regulator will be willing to purchase it with an added fixed incentive bonus. We assume there is no good long term energy storage on the market, so all trades must be closed on a minute by minute basis.

The objective of the challenge is to use the dataset in question to predict, for a given particular minute in the future, given all the data for a year up to the hour before that minute, whether to buy or sell. 


I started with a simple linear regression to see what the naïve result would be, with the idea that an ML result that was widely divergent should be considered suspect.

I then used scikit-learn to implement a simple random forest binary classifier. In the time I had, I only implemented three parameters, the day of the week, to account for weekend/weekday etc variation. The day of the month to attempt to account for holidays, paychecks, months reporting from energy companies etc, and the month of the year, to attempt to account for seasons, weather and climate etc. As I note in the comments, this is too simplistic to be very useful, an accurate model would probably need to account for weather patterns and climate, national holidays, geopolitics and trends in the energy sector, the political agenda of the government and the regulator etc. 

After running for a few seconds, the model made the call to buy, with an expected profit of 2.33 euros per trade. I had then managed to make a very simple start on tuning the model hyperparamters before I ran out of time.

Overall, this mini-project was a very interesting change of subject matter for myself, and although I did not win the challenge (I hadn't quite expected to with my background!) it was a great way to explore the toolkit that I had built for myself as an astronomer in a very different context.

My goals, when I can next find any free time, would be to add some more model parameters and also to tune hyperparams. I don't have a great deal of experience of doing this in a Bayesian way outside of an astrophysics context, so I think that this will be a useful learning experience.
