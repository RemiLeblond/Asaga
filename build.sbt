import com.github.retronym.SbtOneJar._

oneJarSettings

name := "AsyncSaga"

version := "0.1"

scalaVersion := "2.11.6"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.11.2"
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.6" % "test"
libraryDependencies += "com.google.guava" % "guava" % "12.0"

test in assembly := {}

