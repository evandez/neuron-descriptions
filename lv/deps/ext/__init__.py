"""Extensions for third party modules.

That is, code written by the authors of this project for the purposes of
extending the modules found in `third_party` or in `requirements.txt`.

You might ask: why not edit `third_party` directly? After all, you went to
the trouble of copy pasting the third party code into your own repo. Why not
change it?

The answer is that we do not want to fork third party repositories, as this
makes it hard to distinguish bugs from the original third party code vs. our
changes to it. Forking also makes it harder to upgrade the dependency later on.
"""
