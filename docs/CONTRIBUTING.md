# Contribution Guidelines

Please note that this project is released with a
[Contributor Code of Conduct](CODE-OF-CONDUCT.md). By participating in this
project you agree to abide by its terms.  You will also need to sign our
[Contributor License Agreement](RETAIN%20Keras%20Individual%20Contributor%20License%20Agreement%20-%20February%202018.pdf) prior to
submitting any changes to the project. Once completed, the agreement should be
emailed to [opensource@optum.com][email].

---

# How to Contribute

Now that we have the disclaimer out of the way, let's get into how you can be a
part of our project. There are many different ways to contribute.

## Issues

We track our work using Issues in GitHub. Feel free to open up your own issue
to point out areas for improvement or to suggest your own new experiment. If you
are comfortable with signing the waiver linked above and contributing code or
documentation, grab your own issue and start working.

## Coding Standards

We have some general guidelines towards contributing to this project.

### Languages

*Python*

The source code for this project is written in Python. You are welcome to add versions of files for other languages, however the core code will remain in Python.

### Keras Backends

*Tensorflow*

By default we assume that this reimplementation will be run using Tensorflow backend. As Keras grows its support for other backends, we will welcome changes that will make these scripts backend independent.  

## Pull Requests

If you've gotten as far as reading this section, then thank you for your
suggestions.

### General Guidelines

Ensure your pull request (PR) adheres to the following guidelines:

* Try to make the name concise and descriptive.
* Give a good description of the change being made.  Since this is very
subjective, see the [Updating Your Pull Request (PR)](#updating-your-pull-request-pr)
section below for further details.
* Every pull request should be associated with one or more issues.  If no issue
exists yet, please create your own.
* Make sure that all applicable issues are mentioned somewhere in the PR description.  This
can be done by typing # to bring up a list of issues.

### Updating Your Pull Request (PR)

A lot of times, making a PR adhere to the standards above can be difficult.
If the maintainers notice anything that we'd like changed, we'll ask you to
edit your PR before we merge it. This applies to both the content documented
in the PR and the changed contained within the branch being merged.  There's no
need to open a new PR. Just edit the existing one.

[email]: mailto:opensource@optum.com
