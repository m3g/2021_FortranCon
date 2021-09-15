double precision function wrap(x,side)
    implicit none
    double precision :: x, side
    wrap = dmod(x,side)
    if (wrap >= side/2) then
        wrap = wrap - side
    else if (wrap < -side/2) then
        wrap = wrap + side
    end if
end function wrap

subroutine force_pair(fpair,x,y,cutoff,side)
    implicit none
    integer :: i
    double precision :: fpair(2), wrap, d
    double precision :: x(2), y(2), cutoff, side, dv(2)
    do i = 1,2
        dv(i) = wrap(y(i) - x(i), side)
    end do
    d = dsqrt(dv(1)**2 + dv(2)**2)
    if (d > cutoff) then
        fpair(1) = 0.d0
        fpair(2) = 0.d0
    else
        dv = dv / d
        do i = 1,2
           dv(i) = (dv(i)/d)*(d-cutoff)**2
        end do
        fpair = dv
    end if
end subroutine force_pair

subroutine forces(n,f,x,cutoff,side)
    implicit none
    integer :: n, i, j
    double precision :: f(2,n), x(2,n)
    double precision :: fpair(2)
    double precision :: cutoff, side
    do i = 1, n-1
        do j = i+1, n
            call force_pair(fpair,x(:,i),x(:,j),cutoff,side)
            f(1,i) = f(1,i) - fpair(1)
            f(2,j) = f(2,i) + fpair(2)
        end do
    end do
end subroutine forces

double precision function dble_rand()
    call random_number(dble_rand)
end function dble_rand

subroutine md(n,x0,v0,mass,dt,nsteps,isave,trajectory,cutoff,side)
    implicit none
    integer :: n, i, j, k, step, nsteps, isave, isaved
    double precision :: dt
    double precision :: x0(2,n), v0(2,n), mass(n)
    double precision :: x(2,n), v(2,n), f(2,n), a(2,n)
    double precision :: trajectory(2,n,nsteps/isave+1)
    double precision :: cutoff, side

    ! Save initial positions
    trajectory(:,:,1) = x
    x = x0
    v = v0
    isaved = 1
    do step = 1, nsteps
        ! Compute forces
        call forces(n,f,x,cutoff,side)
        ! Update positions and velocities 
        do i = 1, n
           a(:,i) = f(:,i) / mass(i)
           x(:,i) = x(:,i) + v(:,i)*dt + a(:,i)*dt**2/2
           v(:,i) = v(:,i) + a(:,i)*dt
        end do
        ! Save if required
        if (mod(step,isave) == 0) then
            isaved = isaved + 1
            trajectory(:,:,isaved) = x
        end if
    end do
end subroutine md

program main
    implicit none
    integer, parameter :: n = 100
    integer :: i, j, k, nsteps, isave
    double precision :: x0(2,n), v0(2,n), mass(n)
    double precision :: dt
    double precision :: cutoff, side, wrap
    double precision, allocatable :: trajectory(:,:,:)
    double precision :: dble_rand

    ! Initialize positions and velocities
    do i = 1, n
        do j = 1, 2
            x0(j,i) = -50 + 100*dble_rand()
            x0(j,i) = -50 + 100*dble_rand()
            v0(j,i) = -1 + 2*dble_rand()
            v0(j,i) = -1 + 2*dble_rand()
        end do
        mass(i) = 1.d0
    end do

    ! Parameters
    dt = 0.1
    nsteps = 50000
    isave = 1000
    cutoff = 5.
    side = 100.
    allocate(trajectory(2,n,nsteps/isave + 1))

    ! Run simulation
    call md(n,x0,v0,mass,dt,nsteps,isave,trajectory,cutoff,side)

    open(10,file="traj_fortran.xyz")
    k = 0
    do i = 1, nsteps/isave + 1
        k = k + 1
        write(10,*) n
        write(10,*) " step = ", k
        do j = 1, n
            write(10,*) "He", wrap(trajectory(1,j,k),side), wrap(trajectory(2,j,k),side), 0.d0
        end do
    end do
    close(10)

end program main

