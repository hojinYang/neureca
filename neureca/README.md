# <center>Neureca💡 for Conversational Recommender Systems</center>

This repository contains source code for Neureca project. Neureca is a framework for building conversational recommender (ConvRec) systems. It is also an open-source project that helps any ML researchers develop ConvRec system and contribute to key components easily.


- Neureca is still under development stage: the timeline for the demo of verion 0.0X would be around late May - early June.
- At this moment, I set Neureca repository as a personal repo under my github account. I'll transfer this repo to another orgnaization github (either D3M or independent org only for this project) in the future when development has progressed to some extent. 
- Neureca is temporarily named(as an acronym of NEUral REcommender for Conversational Assistant). Please let me know if you have any other good ideas!

```
.
├── demo-toronto
│   ├── application.py
│   ├── data
│   ├── dialogue_manager
│   └── preprocessed
├── neureca
│   ├── app
│   ├── dialogue_manager
│   ├── explanator
│   ├── nlu
│   └── recommender
└── README.md
```

We can see that the main breakdown of the codebase is between neureca and demo-toronto.

The former, neureca, should be thought of as a Python package that we are developing and will eventually deploy in some way.

The latter, demo-toronto, should be thought of as a demo ConvRec project using Neureca api on `yelp-toronto` review dataset. More information about neureca can be found [here](/neureca)