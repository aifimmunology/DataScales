# Icechunk helper tool implementation plan.

## Role

You are a developer for python package, that practices modulation implementation, object oriented classes, and professional implementation.
You will look into Icechunk and be an expert of the roles and correct usage of it.

## summary

You will build a package centered around enableing users to use the icechunk package. The goal is to have commands similar to git,
allowing users to easily create and use icechunk repositories given that they have a zarr.

This package will be used in the command line similar to git, but also as a python library for certain tasks.
The function/command names should be short and match the goal of what its trying to do. Ask any clarifying questions before beginning.

Build some general, coverage test cases for that package.

## Goals of package useable commands

A few of the commands that should be able to be used:

1. Initialize a store with a given zarr path. This will be available in python API. Will be given a zarr path, icechunk outdir

2. Commit - will be available in the python library. Commits the local changes in the icechunk zarr to the branch that is being worked on. They will include a commit messege, this may or may not be an s3 bucket location

3. Tree - similar to git tree, this will show what branches are available in the icechunk, and the commits for those branches. this will be a command line tool

4. Log - log will show the commit hashes for the current branch, this is a command line tool

4. change branch/checkout branch - one function, this will be in the API or command line

5. Cherrypick a commit hash to change the snapshot of the current icechunk 

## notes

This tool will eventually be combined with the other tools in the DataScales, if it has some shared functionality/can use there core functions it should be doing so.